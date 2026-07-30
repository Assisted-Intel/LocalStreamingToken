#!/usr/bin/env python3
"""
Local Streaming Token — persona chain-of-thought pipeline engine (headless).

Executes a persona's ordered steps, passing structured JSON between them. Kept free of
Flask/Ollama specifics: all I/O is injected as callables, so the engine is unit-testable
without a live model and is reused by the chat route (Phase 7) and the Persona Panel
test-run box (Phase 9).

Step types (from persona.xml):
    llm                 — render prompt template, call the model (optionally with a JSON
                          schema for structured output), validate + retry, store output
    knowledge_retrieval — direct query through the persona's knowledge store (no LLM)
    memory_retrieval    — direct query through the persona's memory store (no LLM)

Injected dependencies (all optional except llm_complete):
    llm_complete(model, messages, schema)      -> str          (one-shot completion)
    llm_stream(model, messages)                -> iter[str]    (final step token stream)
    knowledge_search(queries: list[str])       -> list[dict]
    memory_search(queries: list[str])          -> list[dict]
    rewrite_queries(message, history)          -> {"variants":[...], "keywords":[...]}
    emit(event, data)                          -> None         (progress callback → SSE)

Emitted events: step_started, step_output, step_failed, token, run_complete, run_paused.
"""

import re
import time

from .rewrite import _extract_json


class StepStatus:
    """Lifecycle of one pipeline step, mirrored by the UI's step cards."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


_VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_template(template: str, ctx: dict) -> str:
    """Replace {name} tokens present in ctx; leave unknown braces untouched (so JSON
    examples inside a prompt don't blow up like str.format would)."""
    def sub(m):
        key = m.group(1)
        return str(ctx[key]) if key in ctx else m.group(0)
    return _VAR_RE.sub(sub, template or "")


def validate_against_schema(obj, schema: dict):
    """Minimal structural validation: object type + required keys present. Returns an
    error string, or None when valid. (Not full JSON Schema — enough to catch the common
    local-model failure of omitting fields; the schema still constrains generation via
    Ollama's ``format``.)"""
    if schema is None:
        return None
    if schema.get("type") == "object" and not isinstance(obj, dict):
        return "expected a JSON object"
    for key in schema.get("required", []):
        if not isinstance(obj, dict) or key not in obj:
            return f"missing required field '{key}'"
    return None


def format_excerpts(results: list, label: str = "excerpt") -> str:
    """Render retrieved chunks as readable text for injection into a later prompt."""
    if not results:
        return "(none found)"
    out = []
    for i, r in enumerate(results, 1):
        src = r.get("item_id") or r.get("source_id") or ""
        out.append(f"[{i}] ({src}) {(r.get('content') or '').strip()}")
    return "\n\n".join(out)


class PipelineRun:
    """Serializable execution state (per-step + shared context). Sent to the UI and
    persisted alongside the assistant message."""

    def __init__(self, steps: list):
        self.steps = [{
            "id": s.get("id", f"step{i}"),
            "type": s.get("type", "llm"),
            "status": StepStatus.PENDING,
            "output": None,          # dict (llm/JSON), list (retrieval), or str (final)
            "raw": None,             # raw model text when JSON parsing/validation failed
            "error": None,
            "ms": None,
        } for i, s in enumerate(steps)]
        self.status = "pending"       # pending | running | paused | complete | error
        self.final = ""
        self.ctx = {}

    def to_dict(self):
        """JSON-safe snapshot for the SSE payload and for persisting with the message."""
        return {"status": self.status, "final": self.final, "steps": self.steps}


class PipelineEngine:
    """Runs one persona's steps to completion, or pauses on the first failure so the
    user can fix a step and resume from there.

    Every external capability is injected (see the module docstring), so the engine
    has no dependency on Flask, Ollama, or DuckDB and can be tested with plain
    callables. One engine instance holds the state of one run; ``start()`` resets it.
    """

    def __init__(self, persona, *, llm_complete, llm_stream=None,
                 knowledge_search=None, memory_search=None, rewrite_queries=None,
                 emit=None, max_retries=3):
        self.persona = persona or {}
        self.llm_complete = llm_complete
        self.llm_stream = llm_stream
        self.knowledge_search = knowledge_search
        self.memory_search = memory_search
        self.rewrite_queries = rewrite_queries
        self.emit = emit or (lambda ev, data: None)
        self.max_retries = max_retries
        self._defs = list(persona.get("pipeline", []))
        self.run = PipelineRun(self._defs)

    # --------------------------- context ---------------------------

    def _base_ctx(self, user_message: str, history: str) -> dict:
        """The template variables available to every step before any step has run.
        Steps add to this dict as they produce output, so a later prompt can reference
        an earlier step's fields by name. Only the first 5 speaking examples are
        included, to bound the prompt size."""
        prof = self.persona.get("profile", {})
        sp = self.persona.get("speaking", {})
        style_bits = []
        for k in ("tone", "formality", "vocabulary", "quirks"):
            if sp.get(k):
                style_bits.append(f"{k}: {sp[k]}")
        for ex in (sp.get("examples") or [])[:5]:
            style_bits.append(f"Example — user: {ex.get('user','')} | {prof.get('name','')}: {ex.get('reply','')}")
        return {
            "persona_name": prof.get("name", "the persona"),
            "persona_role": prof.get("role", ""),
            "persona_bio": prof.get("bio", ""),
            "user_message": user_message or "",
            "history": history or "",
            "speaking_style": "\n".join(style_bits) or "(no specific style)",
            "knowledge": "(not retrieved yet)",
            "memories": "(not retrieved yet)",
            "draft": "",
        }

    def _model_for(self, step: dict) -> str:
        """Per-step model override, else the persona's chat model, else "" meaning the
        caller substitutes the chat's own model."""
        return step.get("model") or self.persona.get("models", {}).get("chat_model", "") or ""

    def _system_msg(self):
        """The identity system message prepended to every llm step."""
        prof = self.persona.get("profile", {})
        bits = [f"You are {prof.get('name','a persona')}."]
        if prof.get("role"):
            bits.append(f"Role: {prof['role']}.")
        if prof.get("bio"):
            bits.append(prof["bio"])
        return {"role": "system", "content": " ".join(bits)}

    # --------------------------- step execution ---------------------------

    def _resolve_queries(self, key: str, user_message: str, history: str):
        """Queries for a retrieval step: prefer the analyze step's reworded queries in
        ctx; otherwise call the rewrite service so Prompt Reword works in any pipeline."""
        qs = list(self.run.ctx.get(key) or [])
        kws = list(self.run.ctx.get("keywords") or [])
        if not qs:
            if self.rewrite_queries:
                try:
                    rw = self.rewrite_queries(user_message, history)
                    qs = list(rw.get("variants") or [])
                    kws = kws or list(rw.get("keywords") or [])
                except Exception:
                    qs = []
            if not qs and user_message:
                qs = [user_message]
        if kws:
            qs = qs + [" ".join(kws)]
        return [q for q in qs if q]

    def _run_llm_step(self, sdef, sstate, user_message, history, stream_final=False):
        """Execute one llm step; return True on success, False to pause the run.

        Two modes. A step with no schema is free text — streamed token by token when
        it's the last step, so the user sees the answer arrive. A step with a schema
        must return valid JSON: on failure the error is appended to the messages and
        the call is retried up to ``max_retries`` times, since local models routinely
        omit a field on the first attempt. A structured step's fields are merged into
        the shared context, which is how ``analyze`` hands its queries to the
        retrieval steps."""
        model = self._model_for(sdef)
        schema = sdef.get("schema")
        messages = [self._system_msg()]
        if sdef.get("use_history") and history:
            messages.append({"role": "system", "content": f"Conversation so far:\n{history}"})
        prompt = render_template(sdef.get("prompt", ""), self.run.ctx)
        messages.append({"role": "user", "content": prompt})

        # Final free-text step (no schema): stream if we can.
        if schema is None:
            if stream_final and self.llm_stream:
                parts = []
                for tok in self.llm_stream(model, messages):
                    parts.append(tok)
                    self.emit("token", {"text": tok})
                text = "".join(parts).strip()
            else:
                text = (self.llm_complete(model, messages, None) or "").strip()
            sstate["output"] = text
            self.run.ctx[sstate["id"]] = text
            self.run.ctx["draft"] = self.run.ctx.get("draft") or text
            self.run.final = text
            return True

        # Structured step: call with schema, validate, retry with the error appended.
        err = None
        for attempt in range(self.max_retries):
            msgs = list(messages)
            if err:
                msgs.append({"role": "user", "content":
                             f"Your previous reply was invalid: {err}. Reply again with "
                             f"ONLY valid JSON matching the required fields."})
            raw = self.llm_complete(model, msgs, schema) or ""
            obj = _extract_json(raw)
            err = "response was not valid JSON" if obj is None else validate_against_schema(obj, schema)
            if err is None:
                sstate["output"] = obj
                self.run.ctx[sstate["id"]] = obj
                if isinstance(obj, dict):
                    self.run.ctx.update(obj)         # expose fields (e.g. knowledge_queries)
                return True
            sstate["raw"] = raw
        # Exhausted retries → pause for manual fix.
        sstate["status"] = StepStatus.FAILED
        sstate["error"] = err
        return False

    def _run_retrieval_step(self, sdef, sstate, user_message, history):
        """Execute a knowledge or memory retrieval step. No LLM call: the queries come
        from the analyze step (or the rewrite service), and the formatted excerpts are
        placed in ctx under "knowledge"/"memories" for later prompts. Always returns
        True — retrieving nothing is a valid outcome, not a failure."""
        if sdef["type"] == "knowledge_retrieval":
            queries = self._resolve_queries("knowledge_queries", user_message, history)
            results = self.knowledge_search(queries) if self.knowledge_search else []
            self.run.ctx["knowledge"] = format_excerpts(results)
        else:
            queries = self._resolve_queries("memory_queries", user_message, history)
            results = self.memory_search(queries) if self.memory_search else []
            self.run.ctx["memories"] = format_excerpts(results)
        sstate["output"] = {"queries": queries, "results": results}
        return True

    def _execute(self, start_index, user_message, history):
        """Run steps from ``start_index`` to the end, emitting progress as it goes.

        Stops at the first failure and leaves the run "paused" rather than raising, so
        the caller can surface the partial result and let the user edit and resume.
        Any exception from a step is caught and recorded as that step's error."""
        self.run.status = "running"
        n = len(self._defs)
        for i in range(start_index, n):
            sdef, sstate = self._defs[i], self.run.steps[i]
            sstate["status"] = StepStatus.RUNNING
            sstate["error"] = None
            sstate["raw"] = None
            self.emit("step_started", {"index": i, "id": sstate["id"], "type": sstate["type"]})
            t0 = time.time()
            try:
                if sstate["type"] == "llm":
                    okstep = self._run_llm_step(sdef, sstate, user_message, history,
                                                stream_final=(i == n - 1))
                else:
                    okstep = self._run_retrieval_step(sdef, sstate, user_message, history)
            except Exception as e:
                sstate["status"] = StepStatus.FAILED
                sstate["error"] = str(e)
                okstep = False
            sstate["ms"] = int((time.time() - t0) * 1000)
            if not okstep:
                self.run.status = "paused"
                self.emit("step_failed", {"index": i, "id": sstate["id"],
                                          "error": sstate["error"], "raw": sstate["raw"]})
                self.emit("run_paused", {"index": i, "run": self.run.to_dict()})
                return self.run
            sstate["status"] = StepStatus.DONE
            self.emit("step_output", {"index": i, "id": sstate["id"], "output": sstate["output"]})
        self.run.status = "complete"
        self.emit("run_complete", {"run": self.run.to_dict(), "final": self.run.final})
        return self.run

    # --------------------------- public API ---------------------------

    def start(self, user_message: str, history: str = "") -> PipelineRun:
        """Run the whole pipeline from step 0, discarding any previous run. Returns the
        PipelineRun whether it completed or paused — check ``.status``."""
        self.run = PipelineRun(self._defs)
        self.run.ctx = self._base_ctx(user_message, history)
        self._user_message = user_message
        self._history = history
        return self._execute(0, user_message, history)

    def run_from(self, index: int, edited_output=None) -> PipelineRun:
        """Re-run from ``index`` forward. If ``edited_output`` is given it replaces that
        step's output (canonical use: edit the analyze step's reworded queries), then
        downstream steps are invalidated and re-executed. Rebuilds ctx from steps < index."""
        um = getattr(self, "_user_message", "")
        hist = getattr(self, "_history", "")
        self.run.ctx = self._base_ctx(um, hist)
        # Replay context from already-good upstream steps.
        for j in range(index):
            st = self.run.steps[j]
            out = st.get("output")
            if st["type"] == "llm" and isinstance(out, dict):
                self.run.ctx[st["id"]] = out
                self.run.ctx.update(out)
            elif st["type"] == "llm":
                self.run.ctx[st["id"]] = out
                self.run.ctx["draft"] = out
            elif st["type"] in ("knowledge_retrieval", "memory_retrieval") and isinstance(out, dict):
                key = "knowledge" if st["type"] == "knowledge_retrieval" else "memories"
                self.run.ctx[key] = format_excerpts(out.get("results") or [])
        # Apply the edit to the target step, then invalidate it + downstream.
        if edited_output is not None:
            self.run.steps[index]["output"] = edited_output
            if isinstance(edited_output, dict):
                self.run.ctx[self.run.steps[index]["id"]] = edited_output
                self.run.ctx.update(edited_output)
            start = index + 1
        else:
            start = index
        for j in range(start, len(self.run.steps)):
            self.run.steps[j].update({"status": StepStatus.PENDING, "output": None,
                                      "raw": None, "error": None, "ms": None})
        return self._execute(start, um, hist)
