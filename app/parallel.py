#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Multi-server parallel processing engine. Fans a list of work items out across
several LLM servers running at once, multiplexing every token frame back through
a single callback so the caller can forward them over one SSE stream.

Concurrency is thread-based on purpose: the provider adapters (app.providers) are
synchronous blocking generators and Flask runs threaded=True, so one worker thread
per participating server — each running a blocking generation — is the natural fit
(the same pattern api_servers_scan already uses). This mirrors the multi-Ollama
router in the POI Data Enrichment reference program, minus asyncio.

Two distribution modes:
  * "isolation" — item i is pinned to lane (i % L); each lane works its slice
    sequentially. With N items <= N lanes this is exactly one prompt per server,
    all running at the same time.
  * "balanced"  — a shared queue; whichever lane is free pulls the next item
    (maximum throughput; a faster server does more).
"""

import queue as _queue
import threading


def common_models(server_urls, list_models_fn):
    """Return the sorted intersection of installed model names across the given
    servers — the set of models present on *all* of them (for the "Use common
    model" button). Servers that error or report nothing are skipped. Ported from
    the reference program's _update_common_models_dropdown."""
    sets = []
    for url in server_urls:
        try:
            mlist = list_models_fn(url) or []
        except Exception:
            mlist = []
        if mlist:
            sets.append(set(mlist))
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def run_parallel(items, lanes, mode, stop_event, generate_one, emit):
    """Process `items` across server `lanes` on worker threads.

    items       : list of {item_id, chat, search_query, title}
    lanes       : list of {base_url, model, name} — one worker thread each
    mode        : "balanced" | "isolation"
    stop_event  : threading.Event; set to cancel every lane
    generate_one: generator generate_one(chat, search_query, stop_event) yielding
                  (kind, data) tuples (kind in pass_start/status/reasoning/chunk/
                  pass_end/error) — the shared single-generation core.
    emit        : thread-safe callback taking one frame dict. Frames carry
                  event/lane/item_id plus payload. Emits item_start and item_done
                  around each item; item_done carries the final content/reasoning.

    Blocks until all work finishes or stop_event is set.
    """
    lanes = list(lanes or [])
    items = list(items or [])
    if not lanes or not items:
        return

    L = len(lanes)
    if mode == "isolation":
        buckets = [[] for _ in range(L)]
        for i, it in enumerate(items):
            buckets[i % L].append(it)

        def take(lane_idx):
            b = buckets[lane_idx]
            return b.pop(0) if b else None
    else:
        shared_q = _queue.Queue()
        for it in items:
            shared_q.put(it)

        def take(lane_idx):
            try:
                return shared_q.get_nowait()
            except _queue.Empty:
                return None

    def worker(lane_idx):
        lane = lanes[lane_idx]
        while not stop_event.is_set():
            item = take(lane_idx)
            if item is None:
                break
            item_id = item.get("item_id")
            # Bind this item's chat to the lane's server + model.
            chat = dict(item.get("chat") or {})
            chat["server_url"] = lane.get("base_url") or chat.get("server_url")
            if lane.get("model"):
                chat["model"] = lane["model"]
            search_query = item.get("search_query", "")

            emit({"event": "item_start", "lane": lane_idx, "item_id": item_id,
                  "server": lane.get("base_url"), "server_name": lane.get("name"),
                  "model": chat.get("model"), "title": item.get("title", "")})

            content, reasoning, item_images = "", "", []
            try:
                # The item's title (a filename in batch, or prompt label) tags the
                # context-usage frames + history entry for this lane's current item.
                item_label = item.get("title") or item_id
                for kind, data in generate_one(chat, search_query, stop_event,
                                               batch_item_label=item_label):
                    if stop_event.is_set():
                        break
                    frame = {"event": kind, "lane": lane_idx, "item_id": item_id}
                    frame.update(data or {})
                    emit(frame)
                    if kind == "pass_start":
                        content, reasoning, item_images = "", "", []
                    elif kind == "chunk":
                        content += (data or {}).get("content", "")
                    elif kind == "reasoning":
                        reasoning += (data or {}).get("content", "")
                    elif kind == "image":
                        item_images.append(data or {})
                    elif kind == "pass_end":
                        content = (data or {}).get("content", content)
                        reasoning = (data or {}).get("reasoning", reasoning)
            except Exception as e:
                emit({"event": "error", "lane": lane_idx, "item_id": item_id, "message": str(e)})

            emit({"event": "item_done", "lane": lane_idx, "item_id": item_id,
                  "title": item.get("title", ""), "content": content,
                  "reasoning": reasoning, "images": item_images,
                  "stopped": stop_event.is_set()})

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(L)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
