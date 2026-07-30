# Architecture

How Local Streaming Token is put together, for anyone reading or extending the code.
For what it does and how to run it, see [README.md](README.md).

---

## Shape

**Flask on the back, vanilla JavaScript on the front. No bundler, no npm, no build
step.** `static/app.js` is served to the browser exactly as it exists on disk.

This is deliberate. A local tool that handles your private documents should be one
you can read and run without a toolchain: clone, `pip install -r requirements.txt`,
`python main.py`. Every file in `static/` can be opened and understood as-is, and
there is no compiled artifact standing between the source and what runs.

The server is single-user and binds `127.0.0.1` only. It runs `threaded=True` so that
a long streaming response doesn't block other requests or the native-dialog subprocess,
and `use_reloader=False` so it doesn't open two browsers.

---

## Module map

### Entry point

| File | Responsibility |
|---|---|
| [main.py](main.py) | Parse CLI args, find a free port, create the app, open the browser, serve. Also hosts the `--reset-password` / `--forgot-password` flows, which run without starting the server. |

### Web layer

| File | Responsibility |
|---|---|
| [app/server.py](app/server.py) | `create_app()` app factory. ~99 `/api/*` routes, the `before_request` login gate, the `sse()` frame helper, `generate_one()`, and the run registry that makes generations stoppable. |
| [static/index.html](static/index.html) | Whole UI, tab by tab. Section-bannered with HTML comments. |
| [static/app.js](static/app.js) | All client logic, including `streamSSE` (the SSE consumer). Organized by `// ---- section ----` banners. |
| [static/styles.css](static/styles.css) | Styling; the palette is ported from the original wxPython desktop app. |
| [static/login.html](static/login.html) | Self-contained login page with inline CSS (it must render before the app's assets are trusted). |

**Routes stay thin.** They validate input, call into a service module, and stream the
result. Business logic does not live in `server.py`.

### Generation

| File | Responsibility |
|---|---|
| [app/logic.py](app/logic.py) | GUI-free prompt assembly: `build_messages`, `resolve_rag`/`inject_rag`, `resolve_web_search`/`inject_research`. Takes and returns plain dicts, so it's unit-testable without Flask. |
| [app/providers.py](app/providers.py) | Provider adapters behind one interface. `get_client(server)` dispatches on `server["type"]`. |
| [app/parallel.py](app/parallel.py) | Multi-server fan-out. One thread per participating server, all frames multiplexed back through a single callback. |
| [app/evals.py](app/evals.py) | Model-graded evaluation: fill a template from rows, grade responses, parse the JSON tolerantly, aggregate scores. |

### Retrieval

| File | Responsibility |
|---|---|
| [app/rag.py](app/rag.py) | DuckDB vector store. Chunk → embed → retrieve, with vector / BM25 / hybrid-RRF modes. Skips re-embedding when an item's content hash is unchanged, so indexing on every send is cheap. |
| [app/ingest.py](app/ingest.py) | Text extraction from PDF, EPUB, DOCX, TXT, MD. |
| [app/compile.py](app/compile.py) | The explicit "Compile Data" pass — batched, optionally concurrent embedding of changed chunks. |
| [app/rewrite.py](app/rewrite.py) | Query rewriting (one message → 2–3 retrieval queries) and the ✨ Rewrite button. |

### Personas

| File | Responsibility |
|---|---|
| [app/persona.py](app/persona.py) | The persona XML model and `default_pipeline()`. |
| [app/pipeline.py](app/pipeline.py) | The chain-of-thought engine that executes a persona's steps, headless. |
| [app/persona_store.py](app/persona_store.py) | Knowledge and memory stores, namespaced inside the shared RAG DuckDB file. |
| [app/persona_io.py](app/persona_io.py) | Import/export as XML (definition) or `.zip` bundle (definition + sources + memories). |

### Storage, config, security

| File | Responsibility |
|---|---|
| [app/core.py](app/core.py) | The bottom layer: `OllamaClient`, web search, provider presets, branding constants, all filesystem paths, and the encrypting I/O helpers everything else goes through. |
| [app/crypto.py](app/crypto.py) | Encryption at rest. Its module docstring is the spec — read it first. |
| [app/migrate.py](app/migrate.py) | One-time, idempotent pass that encrypts any legacy plaintext on first login. Runs every launch and skips already-encrypted files. |
| [app/store.py](app/store.py) | Thread-safe JSON persistence. `DEFAULT_SETTINGS` lives here. |
| [app/profiles.py](app/profiles.py) | Data profiles, settings profiles, and incognito. |
| [app/context_tracker.py](app/context_tracker.py) | Per-chat context-usage telemetry for the fullness bar. |
| [app/native_dialog.py](app/native_dialog.py) | Native OS file/folder pickers, run as a tkinter subprocess so tk never touches Flask's threads. |
| [app/database/](app/database/) | The Database Processing tab — a self-contained sub-package with [its own README](app/database/README.md). Registers its `/api/db/*` routes via `register_db_routes(app, ctx)`. |

---

## Request flow

A chat send, end to end:

```
browser (fetch / EventSource)
  → @app.route in server.py
      → login gate (before_request)
      → logic.build_messages(chat, ...)
            ├─ resolve_rag  → rag retrieve → inject_rag
            └─ resolve_web_search → core.web_search → inject_research
      → generate_one(chat, search_query, stop_event)
            → providers.get_client(server).chat_stream(model, messages, ...)
      → sse(event, data) frames
  → streamSSE() in app.js → render
```

**`generate_one`** ([app/server.py:904](app/server.py#L904)) is the single generation
loop shared by the `/send` route, the batch runner, and the parallel engine. It yields
`(kind, data)` tuples where `kind` is one of `pass_start`, `status`, `reasoning`,
`chunk`, `pass_end`, `context`, `error`. It deliberately does *not* touch the run
registry or emit `start`/`done` — the caller wraps those, which is what lets the same
loop serve one chat or sixteen parallel lanes.

**`sse(event, data)`** ([app/server.py:149](app/server.py#L149)) formats one frame. Event
names on the wire: `start`, `status`, `chunk`, `context`, `progress`, `gen_progress`,
`model_start`, `model_done`, `row_result`, `file`, `summary`, `error`, `done`.

**Reasoning models.** Adapters surface chain-of-thought as `("reasoning", …)` chunks
separate from `("content", …)`. Models that inline it as `<think>…</think>` in the
content stream are handled by `ThinkSplitter` in `providers.py`, which holds back a
short suffix so a tag split across two network chunks still parses.

**Tool calling stays Ollama-only.** Cloud providers use the app-side web-search path
(research injected as a message) instead, which keeps one code path for the feature.

---

## Encryption at rest

Read [app/crypto.py](app/crypto.py)'s module docstring for the authoritative version.
The shape:

- A random **AES-256 Data Encryption Key (DEK)** encrypts every data file.
- The DEK is **wrapped** by a key derived from the login password with **scrypt**, and
  the wrapped blob is stored in `settings/app_key.enc` along with the Flask session
  secret and the username.
- **Logging in is unwrapping the DEK.** There is no separate password hash — if the
  password is wrong, the unwrap fails. The unwrapped DEK lives in memory only.
- This is why a forgotten password is unrecoverable, and why `--forgot-password` has to
  erase data rather than reset access.

Everything routes through `core.load_json`/`save_json` → `core.read_bytes`/`write_bytes`,
which transparently AES-256-GCM encrypt (header `LSTENC1\n` + nonce + ciphertext). Text
and binary files (persona.xml, memory JSON, sources, audit `.jsonl`) use
`core.read_text`/`write_text`. DuckDB files use DuckDB's own native encryption through
[`core.duckdb_connect`](app/core.py#L998), which attaches with `ENCRYPTION_KEY` and
issues `USE db` so existing bare-table SQL needs no changes.

**Deliberate plaintext exceptions:**

- `data/profiles.json` and `settings/profiles.json` — the profile registries, read at
  boot *before* login, so they cannot be encrypted. They hold names and IDs only. Use
  `core.load_json_plain`/`save_json_plain`.
- `settings/settings.example.json` — a committed template with no real values.
- **All exports.** A persona or library you export is a normal portable file, by design.
  Separate byte builders handle this; `persona_io` decrypts on the way out.

The database connection vault (`db_vault.enc`) has its **own separate password**,
independent of the app login.

**The login gate:** `create_app()` sets `app.secret_key` from the keyfile, and a
`before_request` guard redirects everything to `/login` (401 for `/api/*`) until both
`session["authed"]` and `crypto.is_unlocked()` hold. While locked, the `Store` starts
**empty and does no disk I/O at all**; `_activate_after_login()` populates it on a
successful `POST /api/login`.

---

## Profiles

Two independent axes, each just a folder of files:

| Axis | Location | Holds |
|---|---|---|
| **Data profiles** | `data/profiles/<id>/` | chats, prompts, libraries, evals, personas, database sessions + vault, RAG store |
| **Settings profiles** | `settings/profiles/<id>/` | `settings.json` — providers, API keys, tokens, defaults |

You can pair any data profile with any settings profile — one set of keys across several
separate bodies of work, or vice versa. **Incognito** is a scratch data profile that is
wiped on exit.

The implementation detail worth knowing: **the per-profile paths in `core.py` are module
globals that get reassigned** by `set_active_data_profile()` and
`set_active_settings_profile()` ([app/core.py:113](app/core.py#L113)). Because every
call site dereferences `core.CHATS_FILE` (etc.) *at call time* rather than importing the
name, reassigning the global transparently redirects all reads and writes with no other
changes anywhere. The globals are seeded to the legacy flat locations so imports have
valid values before `create_app()` activates the real profile.

**If you add a new per-profile file, you must add it to `set_active_data_profile()`** —
otherwise it will silently write to the legacy location and leak across profiles.

---

## Conventions

- **Keep Ollama and DuckDB behind service modules** so they can be mocked in tests.
- **No business logic in routes.** If a handler is growing past input validation and a
  service call, the logic belongs in `logic.py`, `rag.py`, or a new module.
- **All LLM calls go through `providers.get_client`**; all embeddings through
  `core.OllamaClient(...).embed`.
- **New retrieval reuses the `resolve_rag` → `_rag_retrieve` → `inject_rag` seam**, which
  the web-search pair (`resolve_web_search` → `inject_research`) mirrors. Following the
  existing shape means the feature composes with isolate-prompts, strict mode, and the
  context tracker for free.
- **Never write paths directly** — always go through the `core.<CONST>` globals, for the
  profile reason above.
- **Never bypass `core.load_json`/`save_json`** — that's what makes encryption at rest
  hold. Writing a data file with plain `open()` silently creates a plaintext hole.
- **Adding a settings key requires two edits:** `DEFAULT_SETTINGS` in
  [app/store.py](app/store.py) *and* the allowlist in `POST /api/settings`. Missing the
  second means the key silently won't persist.
- **Comment style:** module docstrings explain *why* the module exists and list its
  public surface; function docstrings state contracts and invariants, not restatements
  of the signature; `# ---- section ----` banners divide long files.

---

## Adding a provider

1. Add a preset to `PROVIDER_PRESETS` in [app/core.py](app/core.py#L41) — `key`, `label`,
   `type`, `base_url`, `needs_key`. The UI builds its dropdown from this list and
   auto-fills the base URL, so no frontend change is needed.
2. If the service is OpenAI-compatible, set `"type": "openai"` and you are done —
   `OpenAIClient` already covers it (this is how xAI, Gemini, DeepSeek, Groq, Mistral,
   OpenRouter, and LM Studio are supported).
3. If it needs its own SDK or wire format, add an adapter class in
   [app/providers.py](app/providers.py) implementing:
   ```python
   list_models() -> list[str]
   chat_stream(model, messages, options, stop_event, think=False,
               tools=None, tool_executor=None)
       -> generator of (kind, text)      # kind is "reasoning" or "content"
   ```
   then dispatch to it from `get_client(server)`. Honour `stop_event` between chunks so
   the stop button works, and add a `*_FALLBACK_MODELS` list for when the provider's
   model endpoint can't be reached.

---

## Tests

```bash
pytest
```

`pytest` is a development dependency and is intentionally not in `requirements.txt`
(there's no dev-extras section) — `pip install pytest` to run the suite.

Coverage is currently limited to [tests/test_crypto.py](tests/test_crypto.py), which
exercises the DEK wrapping and round-trip encryption. That file is a reasonable model
for new tests: the service modules take plain dicts and paths, so they can be tested
without Flask, a browser, or a live Ollama.
