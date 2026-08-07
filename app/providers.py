#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Provider abstraction. Every chat backend implements the same tiny interface:

    list_models() -> list[str]
    model_capabilities(model) -> list[str]
    chat_stream(model, messages, options, stop_event, think=False,
                tools=None, tool_executor=None) -> generator of (kind, payload)

`chat_stream` yields four kinds of frame:
    ("content",   str)   the answer text
    ("reasoning", str)   chain-of-thought, when ``think`` is set
    ("image",     dict)  {b64, media_type, index} — an image the model produced
    ("usage",     dict)  {prompt_tokens, completion_tokens}, at most once, at the end

Image *input* is accepted by all three adapters: a message may carry an ``images``
list of {media_type, b64} beside its string ``content``, and each adapter reshapes it
into that provider's wire format (see the shapers below). Image *output* is only ever
emitted by OpenAIAdapter, and only by the handful of models that return image parts
through the chat-completions endpoint (Gemini's *-image models, OpenRouter proxies of
them). Ollama and Anthropic have no way to return one, so they never yield the frame.

`get_client(server)` returns the right adapter for server["type"]:
    ollama    -> OllamaAdapter  (local/remote Ollama)
    anthropic -> AnthropicAdapter (official `anthropic` SDK)
    openai    -> OpenAIAdapter  (official `openai` SDK; OpenAI-compatible, covers
                                 OpenAI, xAI/Grok, Gemini, DeepSeek, Groq, etc.)

Every adapter must poll ``stop_event`` between chunks so the app's stop button can
interrupt a generation mid-stream, and must raise RuntimeError (not the SDK's own
exception type) on failure so callers can report errors uniformly.

Cloud providers use the app-side web-search path (research injected as a message),
so native tool-calling stays Ollama-only — ``tools``/``tool_executor`` are accepted
by every adapter for signature compatibility but only honoured by OllamaAdapter.
"""

import re

from .core import OllamaClient, DEFAULT_LOCAL_URL

# Fallback model lists used when a provider's /models endpoint can't be reached.
ANTHROPIC_FALLBACK_MODELS = [
    "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5",
    "claude-opus-4-7", "claude-sonnet-4-6",
]


# --------------------------- inline <think> splitter ---------------------------

_THINK_TAG = re.compile(r"</?(?:think|thinking|reasoning)>", re.IGNORECASE)
_MAX_TAG = len("</reasoning>")  # longest tag we must not split across chunks


class ThinkSplitter:
    """Streaming splitter for reasoning models that inline their chain-of-thought
    as <think>…</think> in the content stream. `feed(text)` returns a list of
    (kind, text) tuples; `flush()` drains the tail. Handles tags split across
    chunk boundaries by holding back a small suffix."""

    def __init__(self):
        self.in_think = False
        self.buf = ""

    def _emit(self, text):
        """Tag ``text`` with the kind implied by the current inside/outside state."""
        return ("reasoning", text) if self.in_think else ("content", text)

    def feed(self, text):
        """Consume one stream chunk and return the (kind, text) tuples it completes.

        Text that might be the start of a tag split across chunk boundaries is held
        back in ``self.buf`` and emitted by a later feed() or by flush()."""
        self.buf += text
        out = []
        while True:
            m = _THINK_TAG.search(self.buf)
            if not m:
                break
            before = self.buf[:m.start()]
            if before:
                out.append(self._emit(before))
            opening = not m.group(0).startswith("</")
            self.in_think = opening
            self.buf = self.buf[m.end():]
        # Emit everything except a possible partial tag at the tail.
        if self.buf:
            safe = self.buf
            hold = ""
            idx = self.buf.rfind("<")
            if idx != -1 and (len(self.buf) - idx) < _MAX_TAG:
                safe, hold = self.buf[:idx], self.buf[idx:]
            if safe:
                out.append(self._emit(safe))
            self.buf = hold
        return out

    def flush(self):
        """Drain the held-back tail at end of stream. Call once after the last feed()."""
        out = []
        if self.buf:
            out.append(self._emit(self.buf))
            self.buf = ""
        return out


# --------------------------- message shaping ---------------------------

# Internally a message carries images as a sibling of its text:
#     {"role": "user", "content": "<text>",
#      "images": [{"media_type": "image/png", "b64": "…"}]}
# `content` stays a string everywhere in the app (the pre-prompt fold, the context
# tracker and the batch templates all rely on it). Each shaper below converts that
# into its provider's own multi-part format, which is the only place those shapes
# should exist. A message with no images produces exactly what it did before images
# were a feature.

def _split_system(messages):
    """Return (system_text, chat_messages) — hoist all system-role entries into a
    single system string and keep only user/assistant turns (for Anthropic).

    Images become ``image`` content blocks placed *before* the text, which is what
    Anthropic recommends when a turn is a question about a picture."""
    system_parts = []
    chat = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        elif role in ("user", "assistant"):
            imgs = m.get("images")
            if imgs:
                parts = [{"type": "image",
                          "source": {"type": "base64",
                                     "media_type": im.get("media_type", "image/png"),
                                     "data": im.get("b64", "")}}
                         for im in imgs if im.get("b64")]
                parts.append({"type": "text", "text": content})
                chat.append({"role": role, "content": parts})
            else:
                chat.append({"role": role, "content": content})
        # tool / tool_calls roles are dropped (cloud path never has them)
    return ("\n\n".join(system_parts), chat)


def _openai_messages(messages):
    """Keep only role+content for system/user/assistant (OpenAI takes system inline).

    Images become ``image_url`` parts carrying a base64 data URL, the form every
    OpenAI-compatible server understands."""
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        content = m.get("content", "")
        imgs = m.get("images") if role in ("user", "assistant") else None
        if not imgs:
            out.append({"role": role, "content": content})
            continue
        parts = [{"type": "text", "text": content}]
        for im in imgs:
            if not im.get("b64"):
                continue
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:{im.get('media_type', 'image/png')};base64,{im['b64']}"}})
        out.append({"role": role, "content": parts})
    return out


def _ollama_messages(messages):
    """Ollama's /api/chat wants images as a bare list of base64 strings alongside the
    text, not as content parts. Messages without images are handed through unchanged
    (same object), so the text-only path costs nothing."""
    out = []
    for m in messages:
        imgs = m.get("images")
        if not imgs:
            out.append(m)
            continue
        msg = dict(m)
        if m.get("role") in ("user", "assistant"):
            msg["images"] = [im["b64"] for im in imgs if im.get("b64")]
            if not msg["images"]:
                msg.pop("images")
        else:
            msg.pop("images", None)
        out.append(msg)
    return out


# --------------------------- Ollama ---------------------------

class OllamaAdapter:
    """Local or remote Ollama. The only adapter that supports native tool calling
    and exact token counts, and the only one where ``num_ctx`` is meaningful (cloud
    providers have a fixed context window we can't set per request)."""

    def __init__(self, base_url):
        self.client = OllamaClient(base_url or DEFAULT_LOCAL_URL)

    def list_models(self):
        """Model names from the server's /api/tags."""
        return self.client.list_models()

    def model_capabilities(self, model):
        """Capability tags from /api/show (e.g. "tools", "vision"), used by the UI
        to decide whether to offer tool calling for this model."""
        return self.client.model_capabilities(model)

    def chat_stream(self, model, messages, options, stop_event, think=False,
                    tools=None, tool_executor=None):
        """Stream a completion. ``options["num_ctx"]`` sets the context window.

        Never yields ``("image", …)``: Ollama's chat endpoint reads images but has no
        way to return one."""
        num_ctx = options.get("num_ctx") or 4096
        # Ollama's think mode returns reasoning in message.thinking (already tagged
        # by OllamaClient). The splitter is a safety net for models that instead
        # inline <think>…</think> in the content stream.
        splitter = ThinkSplitter() if think else None
        for kind, text in self.client.chat_stream(
            model, _ollama_messages(messages), num_ctx, stop_event,
            think=think, tools=tools, tool_executor=tool_executor,
        ):
            if kind == "usage":
                yield ("usage", text)  # exact token counts (dict), pass through
            elif kind == "reasoning":
                yield ("reasoning", text)
            elif splitter:
                yield from splitter.feed(text)
            else:
                yield ("content", text)
        if splitter:
            yield from splitter.flush()


# --------------------------- Anthropic ---------------------------

class AnthropicAdapter:
    """Claude via the official `anthropic` SDK. Anthropic takes the system prompt as
    a separate argument rather than a message, so `_split_system` hoists it out."""

    def __init__(self, api_key, base_url=None):
        # Imported lazily so the app runs without the SDK installed when no
        # Anthropic server is configured.
        import anthropic
        kwargs = {"api_key": api_key} if api_key else {}
        if base_url and base_url not in ("https://api.anthropic.com", "https://api.anthropic.com/"):
            kwargs["base_url"] = base_url
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(**kwargs)

    def list_models(self):
        """Live model list, falling back to a static list when the endpoint is
        unreachable (offline, bad key) so the dropdown is never empty."""
        try:
            return [m.id for m in self.client.models.list()]
        except Exception:
            return list(ANTHROPIC_FALLBACK_MODELS)

    def model_capabilities(self, model):
        """Always empty: capability tags are an Ollama concept."""
        return []

    def chat_stream(self, model, messages, options, stop_event, think=False,
                    tools=None, tool_executor=None):
        """Stream a completion, negotiating the thinking config down through
        ``think_attempts`` because the accepted shape varies by SDK and model
        version. Raises RuntimeError on failure.

        Never yields ``("image", …)``: Claude reads images but does not produce them."""
        system, msgs = _split_system(messages)
        if not msgs:
            msgs = [{"role": "user", "content": system or "Hello"}]
            system = ""
        max_tokens = int(options.get("max_output_tokens") or 16000)

        # Thinking configs to try in order (best-effort across SDK/model versions).
        if think:
            budget = max(1024, min(max_tokens - 1, 8000))
            think_attempts = [
                {"type": "adaptive", "display": "summarized"},
                {"type": "enabled", "budget_tokens": budget},
                None,
            ]
        else:
            think_attempts = [None]

        # Dict rather than a bare bool so the nested run() can mutate it (and so the
        # retry loop below can tell "this config was rejected outright" apart from
        # "this config worked and then the stream broke half-way").
        emitted = {"any": False}

        def run(think_cfg):
            """One full streaming attempt with a given thinking config."""
            kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs)
            if system:
                kwargs["system"] = system
            if think_cfg:
                kwargs["thinking"] = think_cfg
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if stop_event.is_set():
                        return
                    if getattr(event, "type", "") != "content_block_delta":
                        continue
                    d = getattr(event, "delta", None)
                    dt = getattr(d, "type", "")
                    if dt == "thinking_delta":
                        t = getattr(d, "thinking", "") or ""
                        if t:
                            emitted["any"] = True
                            yield ("reasoning", t)
                    elif dt == "text_delta":
                        t = getattr(d, "text", "") or ""
                        if t:
                            emitted["any"] = True
                            yield ("content", t)
                final = stream.get_final_message()
            usage = getattr(final, "usage", None)
            if usage is not None:
                yield ("usage", {
                    "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
                })
            if not emitted["any"] and getattr(final, "stop_reason", None) == "refusal":
                yield ("content", "[The model declined to respond to this request.]")

        for i, cfg in enumerate(think_attempts):
            try:
                yield from run(cfg)
                return
            except Exception as e:
                # Retry the next thinking config only if nothing was streamed yet.
                if emitted["any"] or i == len(think_attempts) - 1:
                    raise RuntimeError(f"Anthropic error: {e}")


# --------------------------- OpenAI-compatible ---------------------------

_DATA_URL = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.DOTALL)


def _parse_image_part(part):
    """Pull (b64, media_type) out of one returned image part, or (None, None).

    There is no standard for this: image-returning models bolt their own shape onto
    the chat-completions response. Handle the three seen in the wild — a data URL
    under ``image_url`` (Gemini's OpenAI-compat layer, OpenRouter), and raw base64
    under ``b64_json`` or ``data`` (the images-API shape, reused by some proxies).
    """
    if not isinstance(part, dict):
        return None, None
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if isinstance(url, str):
        m = _DATA_URL.match(url.strip())
        if m:
            return m.group(2), m.group(1)
        return None, None
    b64 = part.get("b64_json") or part.get("data")
    if isinstance(b64, str) and b64:
        return b64, (part.get("media_type") or part.get("mime_type") or "image/png")
    return None, None


def _extract_images(obj):
    """Image parts hanging off a delta or message. The `openai` SDK types these
    objects strictly, so a non-standard ``images`` field never becomes an attribute —
    it lands in ``model_extra``. Check both, since local servers return plain dicts."""
    if obj is None:
        return []
    extra = getattr(obj, "model_extra", None)
    parts = None
    if isinstance(extra, dict):
        parts = extra.get("images")
    if parts is None:
        parts = getattr(obj, "images", None)
    if parts is None and isinstance(obj, dict):
        parts = obj.get("images")
    return parts if isinstance(parts, list) else []

class OpenAIAdapter:
    """Any OpenAI-compatible endpoint — OpenAI itself plus xAI/Grok, Gemini,
    DeepSeek, Groq, Mistral, OpenRouter, LM Studio, and custom servers. The api_key
    defaults to a dummy string because local servers require one to be present but
    don't check it."""

    def __init__(self, api_key, base_url):
        # Imported lazily so the app runs without the SDK installed when no
        # OpenAI-compatible server is configured.
        import openai
        self.client = openai.OpenAI(api_key=api_key or "not-needed",
                                    base_url=(base_url or "https://api.openai.com/v1"))

    def list_models(self):
        """Live model list, or empty on failure — many compatible servers don't
        implement /models, in which case the UI lets the user type a name."""
        try:
            return [m.id for m in self.client.models.list()]
        except Exception:
            return []

    def model_capabilities(self, model):
        """Always empty: capability tags are an Ollama concept."""
        return []

    def _create(self, model, msgs, max_tokens, token_param):
        """Open a streaming completion. ``token_param`` names the output-limit
        argument, which differs between model generations (see chat_stream)."""
        kwargs = {"model": model, "messages": msgs, "stream": True, token_param: max_tokens}
        # Request a final usage frame (OpenAI-style). Some OpenAI-compatible servers
        # reject stream_options, so fall back to a plain stream if it errors.
        try:
            return self.client.chat.completions.create(
                stream_options={"include_usage": True}, **kwargs)
        except Exception:
            return self.client.chat.completions.create(**kwargs)

    def chat_stream(self, model, messages, options, stop_event, think=False,
                    tools=None, tool_executor=None):
        """Stream a completion, retrying once with ``max_completion_tokens`` because
        newer reasoning models reject the older ``max_tokens``. Reasoning arrives
        either in a dedicated delta field (DeepSeek-R1 style) or inline as
        <think> tags, so both paths are handled. Raises RuntimeError on failure."""
        msgs = _openai_messages(messages)
        max_tokens = int(options.get("max_output_tokens") or 16000)
        splitter = ThinkSplitter() if think else None
        try:
            try:
                stream = self._create(model, msgs, max_tokens, "max_tokens")
            except Exception as e:
                # Some newer models (o1/gpt-5-style) require max_completion_tokens.
                if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                    stream = self._create(model, msgs, max_tokens, "max_completion_tokens")
                else:
                    raise
            usage_seen = None
            image_index = 0
            for chunk in stream:
                if stop_event.is_set():
                    break
                cu = getattr(chunk, "usage", None)
                if cu is not None:
                    usage_seen = {
                        "prompt_tokens": getattr(cu, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(cu, "completion_tokens", 0) or 0,
                    }
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                # Image-returning models attach their picture to the delta (streaming)
                # or to a final message object, depending on the server.
                for src in (delta, getattr(choices[0], "message", None)):
                    for part in _extract_images(src):
                        b64, media_type = _parse_image_part(part)
                        if b64:
                            yield ("image", {"b64": b64, "media_type": media_type,
                                             "index": image_index})
                            image_index += 1
                # DeepSeek-R1 and similar expose reasoning in a separate field.
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    yield ("reasoning", rc)
                content = getattr(delta, "content", None)
                if content:
                    if splitter:
                        yield from splitter.feed(content)
                    else:
                        yield ("content", content)
            if splitter:
                yield from splitter.flush()
            if usage_seen is not None:
                yield ("usage", usage_seen)
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible error: {e}")


# --------------------------- factory ---------------------------

def get_client(server: dict):
    """Return the adapter for a resolved server entry {type, base_url, api_key}.

    Unknown types fall back to Ollama, so a settings file written by a newer version
    degrades to the local server rather than erroring. Adapters are cheap and hold no
    connection state, so callers construct one per request rather than caching."""
    stype = (server.get("type") or "ollama").lower()
    base_url = server.get("base_url") or ""
    api_key = server.get("api_key") or ""
    if stype == "anthropic":
        return AnthropicAdapter(api_key, base_url)
    if stype == "openai":
        return OpenAIAdapter(api_key, base_url)
    return OllamaAdapter(base_url)
