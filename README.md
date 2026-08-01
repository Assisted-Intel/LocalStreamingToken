# Local Streaming Token

**A local-first LLM workbench that runs entirely on your own machine.**

One command starts a small web server on `127.0.0.1` and opens your browser to it.
From there you get chat, document retrieval, personas, batch processing, model
evaluation, and AI-assisted database editing — all backed by models running on your
own hardware via [Ollama](https://ollama.com), with everything you feed it stored
encrypted on your own disk.

*by Assisted Intel · v2.0.0 · MIT licensed*

---

## Why local-first

The point of this project is to **process information with local models instead of
depending on a hosted provider**. That choice buys three things:

**Your data stays yours.** Documents, chats, personas, and database rows are read
from and written to your disk. Nothing is uploaded. When you pick a folder or a file,
a native OS dialog runs and only the *path* crosses the wire to the local server —
file contents never travel through the browser. Everything the app persists under
`data/` is AES-256-GCM encrypted at rest.

**Compute is free, so quality gets expensive.** When you aren't paying per token, it
stops being wasteful to run a query three different ways, have a model rewrite your
question before searching, or spend an extra LLM call per chunk at index time. Most of
the features below trade time for a better answer — a trade that only makes sense when
the tokens are yours. They're individually toggleable, so you decide where to spend.

**It works offline.** After install there is no required network dependency. All
runtime dependencies are pure-Python or ship binary wheels, so the whole thing
vendors for an air-gapped machine:

```bash
pip download -r requirements.txt -d wheelhouse     # on a networked machine
pip install --no-index --find-links wheelhouse -r requirements.txt   # on the target
```

Cloud providers (Anthropic, OpenAI, and others) are supported as an **option**, not a
dependency. You can run a local model for retrieval and embeddings while sending one
hard question to a frontier model, and nothing about the app assumes an API key exists.

---

## Spending compute for quality

These are the features that exist specifically because local inference is unmetered.

### Hybrid retrieval with rank fusion
Retrieval runs in one of three modes (**Settings → Retrieval mode**):

| Mode | What it does |
|---|---|
| `hybrid` *(default)* | Runs both vector and BM25 keyword search, merges by reciprocal rank fusion |
| `vector` | Semantic similarity only |
| `keyword` | BM25 only — needs no embedding model at all |

Hybrid does roughly twice the search work per query and reliably beats either half on
its own, because semantic and lexical search fail on different things.

### Multi-query rewriting
Before retrieving, your message is rewritten into 2–3 distinct retrieval queries
(`rag_query_rewrite`, on by default). One LLM call up front, several searches, results
fused — it recovers documents that a literal search of your phrasing would miss.

### Contextual chunking
Optional (`rag_contextual_chunking`, **off** by default). Before a chunk is embedded,
a model writes a sentence situating it inside its parent document, and that sentence is
prepended to the text being indexed. This costs **one LLM call per new or changed
chunk**, which is exactly why it ships off — but it meaningfully improves retrieval on
long documents where a chunk in isolation is ambiguous.

### Persona chain-of-thought pipelines
A persona answers through a multi-step pipeline instead of a single completion. The
default chain is:

```
analyze → retrieve_knowledge → retrieve_memories → synthesize → stylize
```

Each step is separately configurable (its own prompt, model, JSON schema, and whether
it sees conversation history). Every step streams to the UI as a collapsible card you
can expand, inspect, **edit**, and **re-run from that point** — so when an answer goes
wrong you can see which step went wrong and fix it there rather than re-rolling the
whole response. Five model calls where one would do, in exchange for a traceable answer.

### Multi-server parallel fan-out
Point the app at several Ollama servers on your LAN and run work across all of them at
once. Two distribution modes:

- **balanced** — a shared queue; whichever server is free pulls the next item, so
  faster machines do more work
- **isolation** — one prompt pinned per server, all running simultaneously

Results stream back side by side in their own lanes.

### Multi-pass evaluation
Fill a prompt template from spreadsheet rows, run every row, then have a grader model
score each response against criteria you define and return structured JSON. Aggregated
into per-criterion and overall scores — so "did my prompt change help?" is a measurement
rather than a vibe.

### Prompt rewriting
The **✨ Rewrite** button rewrites your message into a clearer instruction before you
send it, with **↶ Undo**. Uses your chat model, or a dedicated rewrite model.

---

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com)** running locally or on your LAN — see
  [OLLAMA_REQUIREMENTS.md](OLLAMA_REQUIREMENTS.md) for install, model choices, and
  GPU/VRAM guidance
- **An embedding model**, for vector and hybrid retrieval:
  ```bash
  ollama pull nomic-embed-text
  ```
  (Not needed if you use `keyword` retrieval mode.) See
  [rag-deps.md](rag-deps.md) for the full list of RAG models, alternatives with their
  dimensions, and which host each one has to be pulled onto.
- **tkinter** — bundled with the standard CPython installer on Windows and macOS.
  On Debian/Ubuntu: `sudo apt install python3-tk`
- *Optional:* `playwright install chromium` for JavaScript-heavy web-search pages
- *Optional:* API keys for any cloud providers you want to add

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

That's it. `main.py` binds **`127.0.0.1` only** (never `0.0.0.0`), takes port **8756**
or the next free one if that's taken, prints the URL, and opens your default browser
after a second. Press **Ctrl+C** in the terminal to stop.

A fresh clone contains no configuration and no data. Everything — the `data/` folder,
the encryption keyfile, your first profile — is created on that first run.

### First login

> **The app ships with the login `admin` / `admin`.**
>
> Sign in and change it immediately — the login page will nag you until you do.

This is not cosmetic. Your login password is what protects your encrypted data:

- **`python main.py --reset-password`** — change the username/password while knowing
  the current one. The encryption key is re-wrapped, so all your data is preserved.
- **`python main.py --forgot-password`** — the last resort. Because the encryption key
  is wrapped by your password and stored nowhere else, a forgotten password is
  **unrecoverable**. This command **erases the encrypted data** and resets the login to
  `admin`/`admin`. It makes you type `ERASE` to confirm. Anything you previously
  exported is unaffected.

---

## Configuring providers

The normal route is the **⚙ Settings** tab in the app — nothing needs to be edited by
hand. [`settings/settings.example.json`](settings/settings.example.json) is committed as
a reference for the file format; your real `settings/settings.json` is gitignored and
encrypted.

Add a server row, pick a type, and the base URL fills itself in:

| Type | Covers |
|---|---|
| Ollama | Local (always available) or any remote Ollama on your LAN |
| Anthropic | Claude, via the official `anthropic` SDK |
| OpenAI | GPT, via the official `openai` SDK |
| xAI / Gemini / DeepSeek / Groq / Mistral / OpenRouter | OpenAI-compatible endpoints |
| LM Studio | Local, `http://localhost:1234/v1` |
| Custom | Any other OpenAI-compatible service |

**Scan for Ollama servers** discovers them across an IP range (`192.168.1.0/24` or
`192.168.1.10-50`); tick the ones you want to add. Each chat remembers its own server
and model.

> **On remote Ollama:** to reach a machine over the LAN you set `OLLAMA_HOST=0.0.0.0:11434`
> there and open TCP 11434. **Only do this on a trusted network** — Ollama has no
> authentication of its own, so anything that can reach the port can use the models.

Web search needs a **Brave** token (discovery) and a **Bright Data** token (page
crawling), both entered in Settings. You can optionally restrict crawling to an
approved-domain list.

---

## What else is in here

**Chats & profiles** — saved and private (unsaved) chats, chat groups, rename/clear/delete,
copy on every message, regenerate the last reply with a different model. Per-chat
pre-prompt (as a system message or folded into the user turn) with reusable presets,
per-chat context length, "isolate prompts" mode that drops history, and reasoning-model
support that separates chain-of-thought from the answer. **Data profiles** keep separate
worlds of chats and libraries; **settings profiles** keep separate sets of providers and
keys; **incognito** leaves nothing behind. A live context-usage bar shows how full the
window is.

**Reference libraries** — build a corpus from typed sections, imported text files, URLs,
or Brave search results. Ingests **PDF, EPUB, DOCX, TXT, and MD**. Import/export as XML.
Select libraries per chat, optionally in **Strict** mode (answer only from these). RAG
chunks and embeds them so large references stay inside your context window, with an
**Auto-RAG** toggle that engages on its own once a message plus attached data crosses a
word threshold.

**Web search** — Brave for discovery, Bright Data for crawling live pages, injected as
research context. Available per message or as a tool the model can call itself.

**Database Processing** — connect to Postgres, MySQL/MariaDB, or SQLite; import rows
into an isolated local DuckDB staging copy; edit and AI-enrich them; preview the exact
UPDATE statements; detect whether the source changed underneath you; then write back
with a full audit log. **The source database is only ever opened writable during
write-back.** See [app/database/README.md](app/database/README.md).

**Batch processing** — point at a folder of `.txt`/`.md` prompts; answers are written to
a `responses/` subfolder beside them.

**Personas** — see above; portable as XML (definition) or a `.zip` bundle (definition +
source documents + memories). Importing a bundle re-ingests and re-embeds locally with
*your* embedding model.

---

## Your data and privacy

- **Encrypted at rest.** A random AES-256 Data Encryption Key encrypts every data file.
  That key is wrapped by a key derived from your login password with scrypt and stored
  in `settings/app_key.enc`. Logging in *is* unwrapping the key — there's no separate
  password hash — and the unwrapped key exists only in memory. DuckDB stores use
  DuckDB's own native encryption.
- **One deliberate exception: the LanceDB vector store.** RAG can use either of two
  vector stores (*Settings → RAG → Vector store*). **LanceDB is the default and is not
  encrypted** — the chunk text and embeddings of everything you compile sit in plain
  files under your data profile, readable without your login password. It is chosen as
  the default because it is dramatically faster: measured at 50k chunks, writes are
  ~155x quicker and keyword search ~174x quicker, because it can use durable on-disk
  vector and full-text indexes that cannot operate on ciphertext. If you would rather
  have the encryption, switch to the **DuckDB** store — it stays fully supported, keeps
  its data, and you can switch back at any time. Both stores are gitignored.
- **Nothing is uploaded.** The server is the only thing that touches your filesystem,
  it's bound to localhost, and file dialogs pass paths rather than contents.
- **Exports are deliberately plaintext**, so a persona or library you export is a normal
  portable file.
- **The repo is safe to fork.** `data/` and everything in `settings/` except the example
  template are gitignored, so your keys and chats can't be committed by accident.

Bear in mind what this is: a **single-user tool bound to localhost**. It trusts its own
operator — SQL fragments in the database tab, for example, are passed through verbatim
by design. Don't expose it to a network you don't control.

---

## Project layout

```
main.py                     Launcher: pick a port, start the server, open the browser
requirements.txt            Python dependencies (annotated, air-gap friendly)
OLLAMA_REQUIREMENTS.md      Ollama install + model/hardware guidance
rag-deps.md                 Which Ollama models RAG needs, and on which host
ARCHITECTURE.md             How it works inside — start here to contribute

app/
  server.py                 Flask app factory, ~99 /api/* routes, SSE, login gate
  logic.py                  GUI-free prompt assembly (messages, RAG, research)
  providers.py              Provider adapters behind one chat_stream() interface
  core.py                   Ollama client, web search, paths, branding, encrypted I/O
  crypto.py                 AES-256-GCM at rest; scrypt-wrapped data key
  migrate.py                One-time, idempotent encryption of legacy plaintext
  store.py                  Thread-safe JSON persistence per profile
  profiles.py               Data profiles + settings profiles + incognito
  rag.py                    DuckDB vector store: chunk, embed, retrieve
  ingest.py                 PDF / EPUB / DOCX / TXT / MD extraction
  compile.py                Batch chunk+embed ("Compile Data")
  rewrite.py                Query rewriting and the Rewrite button
  parallel.py               Multi-server fan-out (thread per lane)
  evals.py                  Model-graded evaluation
  pipeline.py               Persona chain-of-thought engine
  persona.py                Persona XML model
  persona_store.py          Namespaced knowledge + memory stores
  persona_io.py             Persona import / export (XML and .zip bundles)
  context_tracker.py        Per-chat context-usage telemetry
  native_dialog.py          Native OS file/folder dialogs via tkinter
  database/                 Database Processing tab (own README)

static/                     index.html, app.js, styles.css, login.html — no build step
settings/                   settings.example.json (committed); everything else ignored
data/                       Created on first run; encrypted; gitignored
tests/                      pytest
```

---

## Troubleshooting

**No models in the dropdown / "connection refused"** — Ollama isn't running or isn't
reachable. Check `ollama list` locally, or the base URL in Settings for a remote server.
Remote servers need `OLLAMA_HOST=0.0.0.0:11434` set on the *remote* machine.

**RAG returns nothing, or errors about embeddings** — pull the embedding model
(`ollama pull nomic-embed-text`) and confirm **Settings → RAG** points at a server that
has it. Or switch retrieval mode to `keyword`, which needs no embeddings.

**Port 8756 in use** — nothing to do; the launcher falls back to a free port and prints
the URL it chose.

**`ModuleNotFoundError: No module named 'tkinter'`** — install your platform's tk
package (`sudo apt install python3-tk` on Debian/Ubuntu). It's stdlib but packaged
separately on Linux.

**Browser didn't open** — the URL is printed in the terminal; open it manually.

**Forgot the login password** — see [First login](#first-login). It cannot be recovered;
`--forgot-password` erases the encrypted data and starts over.

---

## Contributing

[ARCHITECTURE.md](ARCHITECTURE.md) covers the module map, request flow, encryption
design, and the conventions to follow. Tests run with `pytest`.

## License

MIT — see [LICENSE](LICENSE).
