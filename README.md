# Local Streaming Token

**A local-first LLM workbench that runs entirely on your own machine.**

One command starts a small web server on `127.0.0.1` and opens your browser to it.
From there you get chat, document retrieval, personas, batch processing, model
evaluation, and AI-assisted database editing — all backed by models running on your
own hardware via [Ollama](https://ollama.com), with everything you feed it stored
encrypted on your own disk.

*by Assisted Intel · v2.1.0 · MIT licensed*

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

### Traceable sources
Every RAG answer carries a collapsible **📚 Sources** panel under the reasoning, listing
the chunks that were actually put in front of the model — the document each came from,
its retrieval score, and the excerpt itself. Click one and the app takes you to it: to
the Resources tab with the owning library open and the passage highlighted inside the
item, to the attachment viewer, or to the earlier turn it was quoted from. The panel is
saved with the chat, so you can still check where an answer came from days later.

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

A persona also carries its own **temperature**, its own **speaking style** (tone,
formality, vocabulary, quirks, few-shot examples), and any number of named **style
variants** you can switch between per message from the composer. The persona run stays
isolated from the chat's libraries, attachments, and web search — its own knowledge base
is the point — but it does honor the chat's system prompt. Queue and Batch answer
normally, without a persona.

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
copy on every message, **edit** any message in place, regenerate the last reply with a
different model. Editing works on your own turns and on the assistant's: the text is
replaced, later messages are left alone and nothing regenerates, and your version is what
the model sees on the next send — which makes it the simplest way to steer a conversation
that has drifted. (Ctrl+Enter saves, Esc cancels. In "isolate prompts" mode only the last
message is sent, so an edit to an earlier one won't reach the model — the app says so when
that happens.) Per-chat
pre-prompt (as a system message or folded into the user turn) with reusable presets,
per-chat context length, "isolate prompts" mode that drops history, and reasoning-model
support that separates chain-of-thought from the answer. **Data profiles** keep separate
worlds of chats and libraries; **settings profiles** keep separate sets of providers and
keys; **incognito** leaves nothing behind. A live context-usage bar shows how full the
window is.

**Reference libraries** — build a corpus from typed sections, imported text files, URLs,
YouTube videos, RSS/podcast feeds, audio and video files, or Brave search results.
Ingests **PDF, EPUB, DOCX, TXT, and MD**.
Import/export as XML. Select libraries per chat, optionally in **Strict** mode (answer
only from these). RAG chunks and embeds them so large references stay inside your
context window, with an **Auto-RAG** toggle that engages on its own once a message plus
attached data crosses a word threshold.

**Chat attachments** — the composer's **＋ Add** button offers the same sources
without creating a library, for material that belongs to one conversation. Each addition
becomes a chip that Send consumes; 📌 pins it to the chat instead, so it stays attached
and is put in front of the model on every turn until you remove it.

**RAG scope** — besides your selected libraries, the **Scope** control decides what RAG
searches: *Attachments + data* (what you attached to the chat), *Conversation* (the chat
itself), or *Both*. Searching the conversation is how a long chat outlives its context
window — only the last few turns are sent verbatim (*Settings → RAG → thread window*)
and older ones come back as retrieved excerpts when they're relevant, instead of being
cut off the front. Indexing is incremental and happens as you send, so an unchanged chat
costs nothing; editing a message re-embeds only that message, and regenerating drops the
old answer from the index. **Private chats are never indexed** — the default vector store
is plaintext on disk — so they retrieve in memory each send instead. Clearing a chat, or
deleting it or its tab, removes what was indexed.

**Images** — attach pictures to a chat by picker, by dragging them onto the composer, or
with Ctrl+V straight from the clipboard, and ask a vision model to describe, read, or
compare them. Formats no model accepts (TIFF, BMP, HEIC, …) are converted first, EXIF
rotation is applied so a phone photo isn't read sideways, and images are scaled to a
sensible size before sending — 🖼 **Full res** per chat when the detail matters. The
topbar says whether the selected model reads images at all. Models that can *return* a
picture (Gemini's image models, and OpenRouter proxies of them) have theirs rendered in
the answer and savable to disk. The Batch tab takes a **folder of images** as its input,
one item per picture, and can send **reference images** alongside every item; anything
the model draws is exported next to the text under the same naming rules.

**YouTube** — paste a video URL to pull its transcript and comments into one document,
with the title, channel, date and view count, and per-comment like counts. Comments are
optional and capped by a setting. Fetching goes through Bright Data if you have a token,
otherwise straight from this machine; install the optional `yt-dlp` package and it acts
as a third fallback — worth having, because YouTube currently refuses server-side caption
requests that can't present a proof-of-origin token, and yt-dlp is what fills that gap.

Paste a **playlist** URL into the chat's ＋ Add → ▶ YouTube box and every video is
fetched as its **own attachment**, so you can drop the ones you don't want before
sending. A "Max videos" field appears (0 = all), and Cancel actually stops the run
rather than just closing the panel. A `watch?v=…&list=…` link is genuinely ambiguous —
YouTube's share sheet produces it constantly — so the panel asks whether you meant the
one video or the whole playlist, defaulting to the video.

The **Resources** tab's ▶ Add YouTube button works the same way, with the same
video/playlist prompt and Max-videos field. There each video becomes its **own library
item**, labelled with its title and linking back to its watch URL, and items are saved
as they arrive — cancelling a forty-video playlist at video thirty keeps the thirty you
already have.

**Every fetched video is cached on disk**, per data profile, so the same video is never
crawled twice — a Batch **Preview** and the **Run** that follows it now cost one crawl
instead of two, and re-running a saved batch project is close to instant. Transcripts
and comments are cached separately, so asking for more comments later re-fetches only
the comments and keeps the transcript. Entries **never expire**: tick **Refresh (ignore
cache)** — in the chat panel, the Resources panel, or the Batch tab — to force a fresh
pull, or clear the lot from **Settings → YouTube cache**, which shows how much is stored.
Nothing partial is ever cached: a cancelled run's truncated comment list and a transcript
that failed to load are both left out, so a bad fetch can't harden into a permanent
answer.

**RSS & podcasts** — paste any RSS or Atom feed and pull its items in as documents, from
the chat composer, the Resources tab, the Batch tab, or straight into a persona's
knowledge base. You choose how many of the newest items to take; already-fetched ones
come from cache, so raising that number is how you pick up what's new.

For podcasts this is **Podcasting 2.0**-aware. When a feed publishes a
`<podcast:transcript>` — as the No Agenda feed does, for all 226 of its episodes — the
transcript is downloaded and used directly, which is fast and free. Multiple formats are
tried best-first (the podcast-index JSON, then VTT, then SRT, then plain text), so a dead
link on one falls through to the next rather than giving up. Speaker labels are kept where
the source has them and turned into paragraphs; timestamps are dropped, which removes
about 40% of an SRT file without losing a word of speech. Chapters, hosts and show notes
come along in the same document, above the transcript so they survive if a very long
episode has to be truncated. For a plain blog feed there's no transcript to find, so a
full-text item is used as-is and a summary-only one triggers a fetch of the linked page.

**Local transcription** — for episodes whose feed publishes nothing, tick **Transcribe
missing episodes** and the audio is downloaded and run through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) on this machine. It is **off
by default** for a reason: it is a hundred-megabyte download and minutes of GPU per
episode, and a published transcript is always preferred when one exists. The same engine
handles audio and video you add directly — 🎙 in the composer or the Resources tab — so
you can drop an `.mp3`, `.m4a`, `.mp4` or `.mkv` in and get a transcript back. The
downloaded audio is deleted the moment it has been transcribed; the transcript is the
thing worth keeping. **Settings → Transcription** picks the model, device and compute
type (default `large-v3` on the GPU, falling back to the CPU on its own if the GPU can't
run it) and can unload the model to free its VRAM. Requires the optional
`faster-whisper` package; without it everything else still works and the checkbox is
simply disabled.

Feeds and episodes are cached per data profile, like YouTube videos. Feed listings are
re-read with a conditional request so checking for new episodes costs almost nothing when
nothing has changed; fetched episodes never expire. A locally transcribed episode is
never overwritten by a re-run — and if the publisher later ships a real transcript, that
one replaces it, never the other way round. **Settings → RSS / Podcast cache** shows what
is stored, including how many episodes were transcribed locally, and offers a cheap
"refresh listings" separately from the destructive clear.

**Web search** — Brave for discovery, Bright Data for crawling live pages, injected as
research context. Available per message or as a tool the model can call itself.

**Database Processing** — connect to Postgres, MySQL/MariaDB, or SQLite; import rows
into an isolated local DuckDB staging copy; edit and AI-enrich them; preview the exact
UPDATE statements; detect whether the source changed underneath you; then write back
with a full audit log. **The source database is only ever opened writable during
write-back.** See [app/database/README.md](app/database/README.md).

**Batch processing** — the **🗂 Batch** tab runs one prompt across many inputs. Mix as
many sources as you like: YouTube videos, a whole **YouTube playlist**, the pages behind
a Brave search, or a folder of documents (optionally including subfolders, reading PDF /
EPUB / DOCX / TXT / MD). Write the instruction once as a template — `{{content}}` is the
item's text, plus `{{title}}`, `{{url}}` and `{{source}}` — and it runs against every
item, with the same system prompts, resource libraries and Multi-Pass refinement the chat
has — including its own **editable evaluation prompt**, so you decide what each
refinement round should actually look for. **Preview items** shows exactly what will run
before you start, and because fetched videos are cached (see YouTube above), previewing a
playlist and then running it costs one crawl rather than two.

Results go to a batch transcript you can promote into a normal chat, and/or to exported
files: one file per item, everything in a single file, or written **next to each file it
processed** (which requires a prefix or an appended suffix, so the AI's answer can never
overwrite your originals). Name files from the source title or have the model write a
title from the content it just produced, with an editable naming prompt. Choose the
folder, the extension, and whether the file also carries the source text and the prompt.
Saved as named **batch projects**, so a routine job is one dropdown away.

The chat composer's older **📂 Batch** button is still there for the other shape of the
job: a folder where each `.txt`/`.md` file *is* a prompt, with answers written to a
`responses/` subfolder beside them.

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
- **Nothing leaves this machine except what you send to a model.** The server is the
  only thing that touches your filesystem and it's bound to localhost. File dialogs pass
  paths rather than contents; images are the one thing the browser uploads, because a
  pasted screenshot has no path — they go to the local server and are stored encrypted
  like everything else.
- **Exports are deliberately plaintext**, so a persona or library you export is a normal
  portable file. A chat export carries its images inline as base64 for the same reason —
  it has to open on another machine — so an export containing photos is an unencrypted
  copy of them. The app says so when it writes one.
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
  server.py                 Flask app factory, ~150 /api/* routes, SSE, login gate
  logic.py                  GUI-free prompt assembly (messages, RAG, research)
  providers.py              Provider adapters behind one chat_stream() interface
  core.py                   Ollama client, web search, paths, branding, encrypted I/O
  crypto.py                 AES-256-GCM at rest; scrypt-wrapped data key
  migrate.py                One-time, idempotent encryption of legacy plaintext
  store.py                  Thread-safe JSON persistence per profile
  profiles.py               Data profiles + settings profiles + incognito
  memory.py                 User memory cores: what the AI learns about you
  rag.py                    RAG core: chunk, embed, retrieve (vector/keyword/hybrid)
  vectorstore/              Two interchangeable stores: LanceDB (default) or DuckDB
  ingest.py                 PDF / EPUB / DOCX / TXT / MD extraction
  images.py                 Image attachments: transcode, EXIF, downscale, store
  compile.py                Batch chunk+embed ("Compile Data")
  batch.py                  Batch tab: sources -> items, filenames, exports
  youtube.py                Video transcripts + comments; playlist enumeration
  youtube_cache.py          Never-expiring per-profile cache of fetched videos
  rss.py                    RSS/Atom feeds + Podcasting 2.0 transcripts
  rss_cache.py              Per-profile cache of feed listings and episodes
  transcribe.py             Local speech-to-text (faster-whisper) + cue cleaning
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
