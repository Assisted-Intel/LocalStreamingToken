#!/usr/bin/env python3
"""
AI processing bridge — the documented interface between the existing LLM pipeline
and the DuckDB staging tables (deliverable #11).

Nothing new is invented here: a staging row is just a ``dict{column: value}``,
which is exactly what ``app.evals.fill_prompt`` already consumes. For each
configured column we:

  * WEB_SOURCE column -> render its ``search_query`` with fill_prompt, call the
    app's ``core.web_search(query, ...)``, and store the crawled text in the cell
    (so later prompt columns can reference it via ``{ThatColumn}``).
  * PROMPT / OUTPUT column -> render ``prompt_template`` with fill_prompt against
    the row, wrap it in a **synthetic chat** dict
    ``{"server_url","model","messages":[{"role":"user","content":filled}],"num_ctx"}``,
    and run it through the app's ``generate_one`` (single) or
    ``parallel.run_parallel`` (fan-out). The accumulated ``chunk`` text becomes the
    cell value.

Columns process in order, and each output is written back into the DuckDB staging
row immediately, so a later column's template can read an earlier column's output.
Writes go through ``StagingManager.update_cell`` (marking ``__dirty`` + capturing
the original for dry-run/audit), so AI output participates in the same safe
write-back path as manual edits.

The heavy collaborators (generate_one, run_parallel, web_search, fill_prompt) are
injected by the Flask route rather than imported here, keeping this module
framework-free and unit-testable.

Phase status: interface in Phase 0; implemented in Phase 5.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional


def make_chat(server_url: str, model: str, filled_prompt: str, num_ctx: int = 4096) -> dict:
    """Build the synthetic single-turn chat that generate_one / run_parallel expect.
    This is the exact contract the existing pipeline already understands."""
    return {
        "server_url": server_url,
        "model": model,
        "messages": [{"role": "user", "content": filled_prompt}],
        "num_ctx": num_ctx,
    }


class StagingProcessor:
    """Runs the configured columns over a session's staging rows using the app's
    own LLM engine. Injected callables keep it decoupled from Flask/providers."""

    def __init__(self, staging, *, fill_prompt: Callable, generate_one: Callable,
                 web_search: Optional[Callable] = None, run_parallel: Optional[Callable] = None):
        self.staging = staging
        self.fill_prompt = fill_prompt
        self.generate_one = generate_one
        self.web_search = web_search
        self.run_parallel = run_parallel

    def process(self, columns: list, *, server_url: str, model: str, num_ctx: int = 4096,
                web_min_pages: int = 5, stop_event=None) -> Iterator[tuple]:
        """Yield ``(kind, data)`` progress tuples (row_start / cell_done / progress /
        error / done) as each staging row's web-source and prompt columns are filled
        and written back to DuckDB.

        ``columns`` is the ordered list of ColumnDef dicts to run (types web_source /
        prompt / output). Processing is sequential per row and columns run in their
        given order, so a later prompt can reference an earlier column's output via a
        ``{Column}`` placeholder. Each result is written straight back through
        StagingManager.update_cell (marking __dirty + capturing the original), so AI
        output flows through the same safe dry-run/write-back path as manual edits."""
        run_cols = [c for c in (columns or [])
                    if c.get("ctype") in ("web_source", "prompt", "output") and c.get("name")]
        rows = list(self.staging.iter_rows())
        total = len(rows)
        yield ("status", {"message": f"Processing {total} row(s) × {len(run_cols)} column(s)…",
                          "total": total})

        for i, row in enumerate(rows):
            if stop_event is not None and stop_event.is_set():
                break
            rowid = row.get("__rowid")
            # Work on a mutable copy so later columns see earlier outputs this row.
            ctx = {k: v for k, v in row.items() if k != "__rowid"}
            yield ("row_start", {"rowid": rowid, "index": i, "total": total})

            for col in run_cols:
                if stop_event is not None and stop_event.is_set():
                    break
                name, ctype = col["name"], col["ctype"]
                try:
                    if ctype == "web_source":
                        if self.web_search is None:
                            raise RuntimeError("Web search is not available.")
                        query = self.fill_prompt(col.get("search_query", ""), ctx).strip()
                        text = ""
                        if query:
                            text = self.web_search(
                                query, min_pages=web_min_pages,
                                should_stop=(lambda: bool(stop_event and stop_event.is_set())),
                                allowed_domains=col.get("domains") or None) or ""
                        value = text[:4000]
                    else:  # prompt / output
                        filled = self.fill_prompt(col.get("prompt_template", ""), ctx,
                                                  col.get("input_columns") or None)
                        chat = make_chat(server_url, model, filled, num_ctx)
                        content = ""
                        for kind, data in self.generate_one(chat, "", stop_event):
                            if kind == "chunk":
                                content += data.get("content", "")
                            elif kind == "error":
                                raise RuntimeError(data.get("message", "generation error"))
                        value = content.strip()

                    self.staging.update_cell(rowid, name, value)
                    ctx[name] = value
                    yield ("cell_done", {"rowid": rowid, "column": name, "ctype": ctype,
                                         "chars": len(value or "")})
                except Exception as e:
                    yield ("error", {"rowid": rowid, "column": name, "message": str(e)})

            yield ("progress", {"done": i + 1, "total": total})

        yield ("done", {"processed": total})
