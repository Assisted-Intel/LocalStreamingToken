# RAG model dependencies — LocalStreamingToken

Which Ollama models the RAG pipeline needs, why, and where each one has to be pulled.

For installing Ollama itself, GPU/RAM sizing and per-platform setup, see
[OLLAMA_REQUIREMENTS.md](OLLAMA_REQUIREMENTS.md). This file covers **models only**.

---

## TL;DR

One model is required. Everything else is optional or reuses a model you already have:

```bash
ollama pull nomic-embed-text
```

That is enough for the full RAG feature set — indexing ("Compile Data"), and vector,
keyword and hybrid retrieval.

---

## What RAG uses, and where

| Purpose | App setting | Runs on | Required? |
|---|---|---|---|
| **Embeddings** — turns chunks and queries into vectors | `rag_embed_model` (default `nomic-embed-text`) | the **embedding** server (`rag_embed_server_url`, plus every host in `rag_embed_servers`) | **Yes**, for `vector` and `hybrid` retrieval |
| **Query rewrite** — expands your message into 2-3 retrieval queries | `rewrite_model`; falls back to the chat's own model | the **chat** server | No — works out of the box with no extra pull |
| **Contextual chunking** — writes a situating sentence for every chunk | `rag_context_model`, else `rewrite_model` | the **embedding** server ⚠️ | No — off by default |

The two gotchas hidden in that table:

- **Contextual chunking runs on the embedding host, not the chat host.** If you enable it
  and your chat model only exists on a different machine, the compile will silently skip
  contextualisation. Pull the context model onto the embedding server.
- **Contextual chunking does not fall back to the chat model** during Compile Data. If
  both `rag_context_model` and `rewrite_model` are empty, it is skipped without an error.

---

## 1. Embedding model — the only hard requirement

**Settings → RAG → Embedding model.** Default: `nomic-embed-text`.

```bash
ollama pull nomic-embed-text          # ~274 MB, 137M params, 768 dimensions
```

It is the default because it is small, fast on CPU, and by far the most widely used
embedding model in the Ollama library — a safe choice you can leave alone.

### Alternatives

All of these are in the official Ollama library (`ollama pull <name>`):

| Model | Size tag | Dimensions | Notes |
|---|---|---|---|
| `nomic-embed-text` | 137m | **768** | Default. Best size/quality balance for most people. |
| `all-minilm` | 22m, 33m | **384** | Tiny and very fast. Good for weak/CPU-only machines; lower retrieval quality. |
| `mxbai-embed-large` | 335m | **1024** | Stronger retrieval, ~2.5x the size and slower to index. |
| `bge-m3` | 567m | probe it | Multilingual, multi-granularity. Pick this for non-English corpora. |
| `embeddinggemma` | 300m | probe it | Google's compact embedding model. |
| `qwen3-embedding` | 0.6b, 4b, 8b | probe it | Strong quality; the larger tags are slow to index a big corpus. |
| `snowflake-arctic-embed` / `-embed2` | 22m–568m | probe it | Wide size range; `2` is the newer multilingual one. |
| `granite-embedding` | 30m, 278m | probe it | IBM; small and permissively licensed. |
| `paraphrase-multilingual` | 278m | probe it | Sentence-similarity oriented, multilingual. |
| `bge-large` | 335m | probe it | Older BGE generation; `bge-m3` generally supersedes it. |

Dimensions marked **bold** are confirmed; for the rest, check yours with the probe below
rather than trusting a table — tags change.

### Check any model's dimensions and context window

```bash
# How many dimensions does it produce?
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["dimension probe"]}' \
  | python -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]), 'dims')"

# What context window does Ollama give it?
curl -s http://127.0.0.1:11434/api/tags \
  | python -c "import sys,json;[print(m['name'], m['details'].get('context_length'), m['details'].get('embedding_length')) for m in json.load(sys.stdin)['models'] if 'embedding' in m.get('capabilities',[])]"
```

### Four rules that will bite you

1. **Changing the embedding model invalidates everything you have compiled.** The model
   is part of the compile signature, so every library and persona goes *Stale* and must
   be recompiled. Choose once, early, on a corpus you care about.
2. **Every server in an embedding pool must have the same model.** Multi-server fan-out
   (*Settings → RAG → Use multiple servers*) sends batches to whichever host is free.
   Vectors from different embedding models are not comparable, so the app uses one model
   for the whole pool by design — pull it on every machine. **Settings → RAG → Check
   servers** verifies this for you.
3. **Keep `rag_chunk_size` inside the model's context window.** The default chunk size is
   512 tokens and `nomic-embed-text` reports a 2048-token window, so the default is
   comfortable. If you raise chunk size past the window, the model silently truncates and
   you lose the tail of every chunk with no error.
4. **Dimensions affect storage.** The LanceDB backend keeps one table per embedding width,
   so switching models creates a new table rather than corrupting the old one. Harmless,
   but it does mean the old vectors linger until you recompile.

---

## 2. Query rewrite (optional, no extra pull)

`rag_query_rewrite` is **on by default**. Before retrieval, your message is expanded into
a few reworded queries plus keywords, which are retrieved independently and fused — so
one badly-phrased question does not sink retrieval.

It uses `rewrite_model` if set, otherwise **the chat model you are already talking to**,
on the chat server. There is nothing extra to install.

Set `rewrite_model` to a small fast model only if you want to keep a large chat model
free, e.g.:

```bash
ollama pull llama3.2:3b      # then set it as "Rewrite model" in Settings
```

---

## 3. Contextual chunking (optional, expensive)

`rag_contextual_chunking` is **off by default**, and should usually stay off.

When on, every chunk gets an LLM-written sentence describing how it fits its document,
which is prepended before embedding. It measurably improves retrieval of chunks that are
meaningless out of context — and it costs **one LLM call per chunk**. On a shelf of
ebooks that is tens of thousands of calls and can turn a minutes-long compile into an
hours-long one. The progress bar accounts for it (it becomes the dominant phase), but
budget the time before enabling it.

If you do want it, pull a small instruct model **onto the embedding server** and set it
as *Contextual-chunking model*:

```bash
ollama pull llama3.2:3b       # or qwen3:4b, gemma3:4b — small and fast is the point
```

---

## 4. Running RAG with no models at all

Set **Retrieval mode → Keyword (BM25)**. Keyword retrieval needs no embedding model and
no LLM: it indexes and searches chunk text directly. Useful when

- you have no GPU and indexing a large corpus with embeddings is impractical,
- your embedding server is temporarily unreachable, or
- you want to try the RAG workflow before committing to a model.

Compiling in this mode stores chunks with empty vectors — deliberately, not as an error —
so you can add embeddings later by pulling a model and recompiling. Retrieval quality is
noticeably lower than hybrid: keyword search finds matching words, not matching meaning.

Note this also interacts with the vector store: on the **LanceDB** backend keyword search
uses a native full-text index and stays fast on large corpora, whereas the **DuckDB**
backend scores BM25 in Python across the whole scope, which slows markedly as the corpus
grows. See *Settings → RAG → Vector store*.

---

## 5. Verify your setup

```bash
# 1. Is Ollama up?
curl http://127.0.0.1:11434/api/tags

# 2. Is an embedding-capable model installed?
ollama list

# 3. Does the embedding model actually return a vector?
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["hello"]}' | head -c 200
```

In the app: **Settings → RAG → Check servers** probes every configured embedding host and
reports whether each is reachable *and* has the embedding model pulled — a server that
answers but lacks the model would otherwise fail batches partway through a compile.

---

## 6. Rough sizing

| | Disk | Practical note |
|---|---|---|
| `all-minilm` | ~46 MB | Runs on anything. |
| `nomic-embed-text` | ~274 MB | Comfortable on CPU-only machines. |
| `mxbai-embed-large` | ~670 MB | Wants a GPU for a large corpus. |
| `bge-m3` | ~1.2 GB | Wants a GPU. |
| Small instruct model (optional, for §2/§3) | 2–3 GB | Only if you set a rewrite/context model. |

Embedding models are small compared with chat models — the corpus, not the model, is what
takes up space. For throughput tuning (`OLLAMA_NUM_PARALLEL`, batch size, per-server
concurrency), see [OLLAMA_REQUIREMENTS.md](OLLAMA_REQUIREMENTS.md#3-recommended-settings-for-faster-compile-data).

---

## Sources

- [Ollama embedding models (library)](https://ollama.com/search?c=embedding)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
- [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large)
- [Ollama blog: embedding models](https://ollama.com/blog/embedding-models)
