# Ollama Requirements — LocalStreamingToken

This app uses a local [Ollama](https://ollama.com) server for two things:

1. **Embeddings** for RAG / "Compile Data" (always local Ollama, independent of
   your chat provider).
2. **Chat generation**, when you pick a local model instead of a cloud provider
   (OpenAI / Anthropic).

You do **not** need a GPU. Ollama runs on NVIDIA, Apple Silicon (Metal), AMD
(ROCm), Intel, or plain **CPU** as a last resort — it picks the best backend
automatically. A GPU mainly makes **Compile Data** faster.

---

## 1. Quick start (all platforms)

```bash
# 1. Install Ollama (see per-platform section below)
# 2. Pull the embedding model the app needs by default:
ollama pull nomic-embed-text

# 3. (Optional) pull a local chat model, e.g.:
ollama pull llama3.1:8b

# 4. Confirm the server is up:
ollama list          # lists installed models
curl http://127.0.0.1:11434/api/tags     # should return JSON, not an error
```

The app talks to Ollama at **`http://127.0.0.1:11434`** by default. If your
Ollama runs elsewhere (another machine, a Windows Server, a container), set the
address in the app under **Settings → RAG → Embed server URL**.

---

## 2. Minimum requirements

| Item | Requirement |
|------|-------------|
| Ollama version | Recent build that supports the batch endpoint `POST /api/embed` (any 2024+ release). Older versions still work but embed **one chunk per HTTP call** — much slower. Update if compiles feel slow. |
| Embedding model | **`nomic-embed-text`** (default). Small, ~275 MB, runs on CPU or any GPU. |
| Chat model (only if using local chat) | Any Ollama model, e.g. `llama3.1:8b`. Size depends on your RAM/VRAM. |
| RAM (CPU-only) | ~2 GB free for `nomic-embed-text`; more for chat models (an 8B chat model wants ~8 GB). |
| Disk | ~300 MB for the embed model; chat models are 2–10 GB+ each. |
| Network | None at runtime — everything is local once models are pulled. |

### Models the app references

- **Embedding (required):** `nomic-embed-text` — the default in
  **Settings → RAG → Embedding model**.
- **Optional embedding alternatives** (change in the same setting):
  - `all-minilm` — smallest/fastest, best for CPU-only machines, lower quality.
  - `snowflake-arctic-embed2` or `mxbai-embed-large` — higher retrieval quality,
    slower and larger.
  - ⚠️ If you change the embed model you must **re-run Compile Data** — embeddings
    from different models are not comparable, and the app will mark libraries/
    personas stale until recompiled.
- **Chat (optional):** any local model you select in the chat UI.

📖 **[rag-deps.md](rag-deps.md)** covers the RAG models in full: the complete list of
embedding models with dimensions, which *host* each auxiliary model must be pulled onto
(contextual chunking runs on the embedding server, not the chat server), how to run RAG
with no models at all, and probe commands for checking a model's dimensions and context
window.

---

## 3. Recommended settings for faster "Compile Data"

The app pools and parallelizes embedding during compile. Two Ollama-side
environment variables let that parallelism actually land:

| Variable | Suggested value | What it does |
|----------|-----------------|--------------|
| `OLLAMA_NUM_PARALLEL` | `4` | Number of requests Ollama serves at once. Let the app's concurrent embed batches run in parallel instead of queueing. On **CPU-only**, leave low (`1`–`2`). |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps the embed model loaded in memory between compiles, avoiding a reload each run. |

Matching app settings (**Settings → RAG**):

- **Embed requests per server** (`rag_embed_concurrency`) = **3** on a GPU (keep it
  **≤ `OLLAMA_NUM_PARALLEL`**), or **1** on a CPU-only machine. This is now per
  server, not per compile.
- **Embed batch size** (`rag_embed_batch_size`) = **64** (chunks per request; fine on
  all hardware).

### Using several machines at once

Tick **"Use multiple servers to build RAG"** and add each Ollama host under
*Settings → RAG*. During a compile, batches are handed to whichever server is free, so
a faster box simply does more; **Check servers** verifies each one is reachable and
actually has the embedding model pulled.

Every server in the pool uses the single **Embedding model** setting. There is no
per-server model on purpose: vectors from different embedding models occupy different
spaces and are not comparable, so mixing them would corrupt the index. Run
`ollama pull nomic-embed-text` (or whichever model you chose) on every machine.

If a server dies mid-compile its batches are retried on another one; if some chunks
still can't be embedded, those documents are reported and left out of the manifest so
they recompile next time rather than being recorded as done.

Also: keep **Contextual chunking OFF** unless you specifically want it — it runs a
full LLM call per chunk and is by far the biggest compile cost when enabled.

### If compiles are inexplicably slow

DuckDB probes for `pandas`/`numpy` while converting values. If one of those is
**installed but broken** (typically a numpy/pandas ABI mismatch — *"numpy.dtype size
changed"*), Python retries the failing import for every value, and indexing slows by
roughly two orders of magnitude. The app detects this at startup, disables the broken
package for its own use, and prints a warning naming it. Fix the environment with:

```
pip install -U --force-reinstall numpy pandas
```

---

## 4. Install & configure per platform

### Windows 11 / 10 (desktop, with GUI)

1. Download and run the installer from <https://ollama.com/download/windows>.
   Ollama installs as a background service and a tray icon; the server starts
   automatically and listens on `127.0.0.1:11434`.
2. Pull models from **PowerShell** or **Command Prompt**:
   ```powershell
   ollama pull nomic-embed-text
   ```
3. **Set environment variables** (so they persist across reboots):
   - Open **Start → "Edit environment variables for your account"**.
   - Under *User variables* click **New** and add:
     - `OLLAMA_NUM_PARALLEL` = `4`
     - `OLLAMA_KEEP_ALIVE` = `30m`
   - **Quit Ollama from the tray icon and reopen it** (or reboot) so the service
     picks up the new variables.
   - Quick check in PowerShell: `Get-ChildItem Env:OLLAMA*`
4. **GPU:** NVIDIA GPUs are used automatically if you have current drivers +
   CUDA-capable hardware. Verify with `ollama ps` while a model is loaded — it
   shows `100% GPU` when offloaded, `100% CPU` when not.

### Windows Server (headless / no tray)

1. Install with the same installer, **or** for a locked-down server use the
   standalone zip from the download page and run `ollama.exe serve`.
2. To run Ollama as a **service** that starts on boot and is reachable by the app:
   - Set machine-wide variables (PowerShell as Administrator):
     ```powershell
     [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "Machine")
     [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "4", "Machine")
     [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "Machine")
     ```
     `OLLAMA_HOST=0.0.0.0:11434` makes Ollama accept connections from other
     machines — only do this on a trusted network, and open the firewall port:
     ```powershell
     New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
     ```
   - Restart the Ollama service (`Restart-Service Ollama` if installed as a
     service, or restart the `ollama serve` process).
3. In the app, set **Settings → RAG → Embed server URL** to
   `http://<server-ip>:11434`.
4. **GPU on Windows Server:** install the vendor GPU driver (NVIDIA data-center or
   GeForce driver). Confirm with `ollama ps`.

### macOS (Apple Silicon or Intel)

1. Install from <https://ollama.com/download/mac> (drag to Applications) **or**
   via Homebrew:
   ```bash
   brew install ollama
   ```
2. Start the server:
   - App version: launch **Ollama.app** (menu-bar icon, auto-starts server).
   - Homebrew/CLI: `ollama serve` (or `brew services start ollama` to run at login).
3. Pull models:
   ```bash
   ollama pull nomic-embed-text
   ```
4. **Set environment variables.** How depends on how you start Ollama:
   - **CLI / Terminal session:** add to `~/.zshrc`:
     ```bash
     export OLLAMA_NUM_PARALLEL=4
     export OLLAMA_KEEP_ALIVE=30m
     ```
     then restart the server.
   - **Ollama.app (launched from Finder):** the app doesn't read your shell rc, so
     register the variables with `launchctl`, then restart the app:
     ```bash
     launchctl setenv OLLAMA_NUM_PARALLEL 4
     launchctl setenv OLLAMA_KEEP_ALIVE 30m
     ```
5. **GPU:** Apple Silicon uses the Metal GPU automatically — nothing to configure.

### Linux

1. Install:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   This sets up a **systemd** service (`ollama.service`) listening on
   `127.0.0.1:11434`.
2. Pull models:
   ```bash
   ollama pull nomic-embed-text
   ```
3. **Set environment variables** for the systemd service:
   ```bash
   sudo systemctl edit ollama.service
   ```
   Add:
   ```ini
   [Service]
   Environment="OLLAMA_NUM_PARALLEL=4"
   Environment="OLLAMA_KEEP_ALIVE=30m"
   # Optional: expose to other machines (trusted networks only)
   # Environment="OLLAMA_HOST=0.0.0.0:11434"
   ```
   Then reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```
   If you run Ollama manually instead (`ollama serve`), just `export` the
   variables in that shell first.
4. **GPU:**
   - **NVIDIA:** install the proprietary driver + CUDA container/toolkit as
     appropriate; the installer detects it. Verify with `nvidia-smi` and
     `ollama ps`.
   - **AMD (ROCm):** install the ROCm-enabled Ollama build per the official Linux
     docs; verify with `ollama ps`.

---

## 5. Verifying it works

```bash
# Server reachable?
curl http://127.0.0.1:11434/api/tags

# Embedding model responds (should return a vector)?
curl http://127.0.0.1:11434/api/embed -d '{"model":"nomic-embed-text","input":"hello"}'

# What's loaded and where (GPU vs CPU)?
ollama ps
```

Then in the app: open **Settings → RAG**, confirm the embed server URL and model,
and click **Compile Data** on a library. Watch that it progresses and that
`ollama ps` shows the embed model resident on your accelerator (or CPU).

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| App says embed server unreachable | Ollama not running, or wrong URL. Start Ollama; check **Settings → RAG → Embed server URL**. For a remote server, set `OLLAMA_HOST=0.0.0.0:11434` and open port 11434. |
| Compile is very slow | Model running on CPU (`ollama ps` shows CPU) → fix GPU driver; **or** old Ollama falling back to one-call-per-chunk → update Ollama; **or** Contextual chunking is on → turn it off. |
| Env vars seem ignored | You set them but didn't **restart Ollama**. On Windows GUI, quit from the tray and relaunch. On macOS app, use `launchctl setenv` then relaunch. On Linux, `systemctl daemon-reload && systemctl restart ollama`. |
| "model not found" | Run `ollama pull nomic-embed-text` (and any chat model you selected). |
| Changed embed model, retrieval got worse / everything stale | Re-run **Compile Data** — embeddings from different models aren't interchangeable. |
| Out of memory on CPU-only box | Use a smaller embed model (`all-minilm`), set `rag_embed_concurrency=1` and `OLLAMA_NUM_PARALLEL=1`. |
