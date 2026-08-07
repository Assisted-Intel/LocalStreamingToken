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
| [app/server.py](app/server.py) | `create_app()` app factory. ~150 `/api/*` routes (about 20 of them registered by `app/database/`), the `before_request` login gate, the `sse()` frame helper, `generate_one()`, and the run registry that makes generations stoppable. |
| [static/index.html](static/index.html) | Whole UI, tab by tab. Section-bannered with HTML comments. |
| [static/app.js](static/app.js) | All client logic, including `streamSSE` (the SSE consumer). Organized by `// ---- section ----` banners. |
| [static/styles.css](static/styles.css) | Styling; the palette is ported from the original wxPython desktop app. |
| [static/login.html](static/login.html) | Self-contained login page with inline CSS (it must render before the app's assets are trusted). |

**Routes stay thin.** They validate input, call into a service module, and stream the
result. Business logic does not live in `server.py`.

### Generation

| File | Responsibility |
|---|---|
| [app/logic.py](app/logic.py) | GUI-free prompt assembly: `build_messages`, `resolve_rag`/`inject_rag`, `resolve_web_search`/`inject_research`, `inject_memory`, `resolve_attachments`/`inject_attachments`. Takes and returns plain dicts, so it's unit-testable without Flask. |
| [app/memory.py](app/memory.py) | User memory cores — what the assistant has learned about the *user*, grown from chats. The core model, the `<user_memory>` render, `resolve_memory`, and the extraction/consolidation prompts + operation applier. Storage-free. |
| [app/providers.py](app/providers.py) | Provider adapters behind one interface. `get_client(server)` dispatches on `server["type"]`. |
| [app/parallel.py](app/parallel.py) | Multi-server fan-out. One thread per participating server, all frames multiplexed back through a single callback. |
| [app/evals.py](app/evals.py) | Model-graded evaluation: fill a template from rows, grade responses, parse the JSON tolerantly, aggregate scores. |
| [app/batch.py](app/batch.py) | The Batch tab's non-LLM half: turning sources into items, filling the `{{content}}` prompt template, choosing and sanitising output filenames, and writing exports. Delegates all ingestion to `ingest`/`youtube`/`core.crawl_search`, and never calls a model or Flask itself. |

### Retrieval

| File | Responsibility |
|---|---|
| [app/rag.py](app/rag.py) | Storage-independent RAG core: chunking, `EmbedPool` (embedding fanned across several Ollama hosts), bounded-wave writes that reuse any vector already computed for identical text, and vector / keyword / hybrid-RRF retrieval. The store itself lives behind `app/vectorstore/`. |
| [app/vectorstore/](app/vectorstore/) | Two interchangeable vector stores behind one interface, chosen in Settings → RAG. `lance_backend.py` — LanceDB, with a durable ANN index and a native full-text (tantivy) index; much faster, but **plaintext on disk**. `duckdb_backend.py` — the original AES-encrypted store, with an in-memory HNSW sidecar and Python BM25. `migrate.py` copies DuckDB → Lance without re-embedding; `scoring.py` holds the shared cosine/BM25/RRF helpers. |
| [app/ingest.py](app/ingest.py) | Text extraction from PDF, EPUB, DOCX, TXT, MD. |
| [app/images.py](app/images.py) | Image attachments: sniffing, EXIF orientation, transcoding formats no model accepts (TIFF/BMP/HEIC/…), the send-time downscale, and the per-profile image store. Bytes live in their own encrypted files rather than inside `chats.json`, which is rewritten in full on every settings keystroke. Pillow is imported lazily, so everything text-only still works without it. |
| [app/youtube.py](app/youtube.py) | A YouTube URL → one plain-text document (transcript + comments). Server-side port of the "YT Copy All" extension. Falls through Bright Data → requests → yt-dlp **per half**: the watch page reliably yields comments, but YouTube answers server-side caption requests with an empty body unless they carry a proof-of-origin token, so the usual outcome is comments from an early rung and the transcript from yt-dlp. `fetch_playlist` enumerates a playlist via yt-dlp's flat extraction (one request, no per-video resolution) and is yt-dlp-only — there is no requests rung, because scraping the playlist page means walking continuation tokens by hand. Note `parse_video_id` ignores `list=` while `parse_playlist_id` reads only it; that asymmetry is deliberate, so a "video in a playlist" URL still means the single video. |
| [app/compile.py](app/compile.py) | The explicit "Compile Data" pass — embeds changed chunks across the server pool, streams phase/progress frames that drive the progress bar and ETA, supports cancellation, and leaves incompletely-embedded items out of the manifest so they recompile. |
| [app/rewrite.py](app/rewrite.py) | Query rewriting (one message → 2–3 retrieval queries) and the ✨ Rewrite button. |

### Personas

| File | Responsibility |
|---|---|
| [app/persona.py](app/persona.py) | The persona XML model and `default_pipeline()`. |
| [app/pipeline.py](app/pipeline.py) | The chain-of-thought engine that executes a persona's steps, headless. |
| [app/persona_store.py](app/persona_store.py) | Knowledge and memory stores, namespaced inside the shared RAG DuckDB file. |
| [app/persona_io.py](app/persona_io.py) | Import/export as XML (definition) or `.zip` bundle (definition + sources + memories). |

Three rules the persona code depends on:

- **The write path validates against the read path.** `PersonaService.save` round-trips
  every definition through `from_xml` before it touches disk, because the two used to
  disagree: saving a persona with a blank name wrote a file nothing could parse, and the
  persona then vanished from the UI while its knowledge base stayed on disk. A folder
  whose XML no longer parses is reported by `list_all` with `broken: True` rather than
  skipped, so it can still be deleted.
- **A run always produces an answer.** Only a free-text llm step assigns `run.final`, so
  `PipelineEngine._resolve_final()` backfills from the last usable step output when a
  pipeline ends in a structured or retrieval step — and when a re-run of the *last* step
  re-executes nothing.
- **The persona path is deliberately isolated from chat context** — no libraries,
  attachments, web search, or memory cores — with one exception: the chat's own system
  prompt is passed in as `extra_system` and appended to the persona's identity message.
  Queue, Batch, and the parallel lanes do **not** carry a persona; the composer warns once
  rather than silently answering without one.

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
            ├─ resolve_web_search → core.web_search → inject_research
            └─ memory.resolve_memory → render_core → inject_memory
      → generate_one(chat, search_query, stop_event)
            → providers.get_client(server).chat_stream(model, messages, ...)
      → sse(event, data) frames
  → streamSSE() in app.js → render
```

**`generate_one`** ([app/server.py:1124](app/server.py#L1124)) is the single generation
loop shared by the `/send` route, the batch runner, and the parallel engine. It yields
`(kind, data)` tuples where `kind` is one of `pass_start`, `status`, `reasoning`,
`chunk`, `image`, `pass_end`, `context`, `error`. It deliberately does *not* touch the run
registry or emit `start`/`done` — the caller wraps those, which is what lets the same
loop serve one chat or sixteen parallel lanes.

**`sse(event, data)`** ([app/server.py:236](app/server.py#L236)) formats one frame. Event
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
[`core.duckdb_connect`](app/core.py#L1119), which attaches with `ENCRYPTION_KEY` and
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
| **Data profiles** | `data/profiles/<id>/` | chats, prompts, libraries, evals, batch projects, personas, memory cores, database sessions + vault, RAG store, attached images |
| **Settings profiles** | `settings/profiles/<id>/` | `settings.json` — providers, API keys, tokens, defaults |

You can pair any data profile with any settings profile — one set of keys across several
separate bodies of work, or vice versa. **Incognito** is a scratch data profile that is
wiped on exit.

The implementation detail worth knowing: **the per-profile paths in `core.py` are module
globals that get reassigned** by `set_active_data_profile()` and
`set_active_settings_profile()` ([app/core.py](app/core.py)). Because every
call site dereferences `core.CHATS_FILE` (etc.) *at call time* rather than importing the
name, reassigning the global transparently redirects all reads and writes with no other
changes anywhere. The globals are seeded to the legacy flat locations so imports have
valid values before `create_app()` activates the real profile.

**If you add a new per-profile file, you must add it to `set_active_data_profile()`** —
otherwise it will silently write to the legacy location and leak across profiles. In
practice that means five edits, four of them mandatory: the constant plus the
`set_active_data_profile()` assignment in [app/core.py](app/core.py); and
`_empty_collections()`, `_load_data_collections()`, `flush_all()` and `merge_into()` in
[app/store.py](app/store.py), alongside a `save_*()` method. `core.DATA_FILE_NAMES` is
the fifth and is *not* one to copy blindly — it drives only the one-time pre-profiles
migration, so a file that never existed in that flat layout (like `batch_projects.json`)
correctly stays out of it.

**A per-profile *directory* needs one more.** `images/` holds files rather than an
in-memory collection, so `_empty_collections`/`_load_data_collections`/`flush_all` have
nothing to do — but `store.merge_into()` does: saving an incognito session out to a real
profile has to copy the referenced files across, or every merged chat ends up pointing at
pictures that were wiped with the scratch. The `personas/` directory is the other case
and predates this note.

---

## Conventions

- **Keep Ollama and DuckDB behind service modules** so they can be mocked in tests.
- **No business logic in routes.** If a handler is growing past input validation and a
  service call, the logic belongs in `logic.py`, `rag.py`, or a new module.
- **All LLM calls go through `providers.get_client`**; all embeddings through
  `core.OllamaClient(...).embed`.
- **New retrieval reuses the `resolve_rag` → `_rag_retrieve` → `inject_rag` seam**, which
  the web-search pair (`resolve_web_search` → `inject_research`) and user memory
  (`memory.resolve_memory` → `inject_memory`) mirror. Following the existing shape means
  the feature composes with isolate-prompts, strict mode, and the context tracker for
  free. Every injector inserts a `system` message immediately before the last user turn,
  and only on pass 0 — refinement passes rework an answer already written with them in view.
- **Two things are called "batch" and they are not the same.** The chat composer's
  📂 Batch button (`POST /api/batch/start`) treats every file in a folder as a *prompt*.
  The **Batch tab** (`POST /api/batch/run`, backed by [app/batch.py](app/batch.py))
  treats an input as *content* that one prompt template runs against. They share no
  code and neither should absorb the other. The tab reaches the model **only** through
  `generate_one`, driven by `parallel.run_parallel` with a single synthetic lane when
  multi-server processing is off — that is what keeps it from growing a second
  generation loop, which is exactly how `/api/batch/start`'s sequential branch drifted
  out of sync with `generate_one` over attachments.
- **Two things are called "memory" and they are not the same.** `persona_store.MemoryService`
  holds a *character's* recollections (emotional weight, narrative time, embedded and
  retrieved per turn). `memory.py` holds a profile of the *user* (plain JSON, edited by
  hand in the Memory tab, injected whole). Neither should grow into the other.
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

1. Add a preset to `PROVIDER_PRESETS` in [app/core.py](app/core.py#L115) — `key`, `label`,
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
