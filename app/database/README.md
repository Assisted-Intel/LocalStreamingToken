# Database Processing tab

Import rows from a real database into an isolated local **DuckDB** staging copy,
edit / AI-enrich them, preview the exact write statements, detect if the source
changed underneath you, then write back (bulk or row-by-row) with a full audit log.
Everything runs on the local machine and works offline after install.

The **source database is only ever opened writable during write-back** — import,
editing and AI processing all happen against the separate DuckDB staging file, so
the source can't change until you approve.

## Module map (`app/database/`)

| Module | Responsibility |
|---|---|
| `models.py` | Dependency-free dataclasses/enums (`ConnectionProfile`, `ColumnDef`, `SelectionSpec`, `ImportSession`, `DryRunResult`, `ConflictReport`, `AuditEntry`) |
| `vault.py` | AES-256-GCM encrypted connection vault; key derived from a master password via scrypt, held in memory only |
| `connections.py` | SQLAlchemy 2.x engine build (URL or discrete fields), SSL args, `test_connection` / `list_tables` / `list_columns`; SQLite opened read-only via `PRAGMA query_only` |
| `types_map.py` | Source SQL type → DuckDB type (preserves DECIMAL precision; VARCHAR fallback) |
| `importer.py` | Row-selection modes (first N / random N / full / custom WHERE) + chunked server-side streaming |
| `staging.py` | One DuckDB file per session; typed table + bookkeeping cols; insert / paged read / cell edit / add column / dirty tracking / fingerprints |
| `processing.py` | **AI bridge** (see below) |
| `dryrun.py` | Render exact UPDATE statements (parameterized + literal), capped, source-column-only |
| `conflict.py` | Diff current source vs import-time fingerprints → unchanged/changed/new/deleted (see *Conflict detection* below) |
| `writeback.py` | Transactional bulk + row-by-row write-back, conflict-guarded, audited |
| `audit.py` | Append-only JSONL audit log per session |
| `routes.py` | All `/api/db/*` Flask routes; registered by `app.server.create_app` via `register_db_routes(app, ctx)` |

Storage: encrypted profiles → `settings/db_vault.enc`; staged data → `data/db/staging/staging_<id>.duckdb`; audit → `data/db/audit/<id>.jsonl`; non-secret session metadata → `data/db_projects.json`.

## How the existing AI pipeline talks to the staging tables

A staged row is just a `dict{column: value}`, which is exactly what the app's
`app.evals.fill_prompt` already consumes. No changes to the LLM engine are needed —
the DB tab reuses `generate_one` (single) exactly as the Chat/Evaluate tabs do.

**Reading** rows for processing:

```python
from app.database.staging import StagingManager
sm = StagingManager(session_id)              # session_id from data/db_projects.json
for row in sm.iter_rows():                    # {"__rowid": int, "<col>": value, ...}
    ...
```

**Filling a column with an LLM** — build a synthetic single-turn chat and run it
through the app's own `generate_one`:

```python
from app import evals
from app.database.processing import make_chat

prompt = evals.fill_prompt("Summarise {name} in one line.", row)   # {Column} placeholders
chat   = make_chat(server_url, model, prompt, num_ctx)             # synthetic chat dict
text   = ""
for kind, data in generate_one(chat, "", stop_event):             # app/server.py:generate_one
    if kind == "chunk":
        text += data["content"]
```

**Writing** the result back into staging (marks the cell dirty + captures the
original for dry-run/audit, so AI output flows through the same safe write-back path
as a manual edit):

```python
sm.add_column("Summary", "VARCHAR")           # once
sm.update_cell(row["__rowid"], "Summary", text)
```

`StagingProcessor` in `processing.py` wraps exactly this loop and also supports
`web_source` columns (which call the app's `core.web_search`). Columns run in the
order given, and each output is written before the next column runs, so a later
prompt template can reference an earlier column via `{ThatColumn}`.

Column roles (`ColumnDef.ctype`): `source` (imported), `web_source`
(`core.web_search(fill_prompt(search_query, row))`), `prompt` / `output` (LLM).

## Conflict detection

At import, each staged row keeps a `__src_key` (its primary-key identity) and a
`__row_hash` (a fingerprint of the source row). A conflict check re-reads the
source, recomputes the hashes, and diffs. **How the source is re-read depends on
the selection mode**, and this matters:

| Selection | Re-read | `changed` / `deleted` | `new` |
|---|---|---|---|
| `full` | one full scan | exact | exact |
| `first_n` / `random_n` / `custom` | keyed: `WHERE pk IN (…)` over the staged keys, batched | exact | **not available** |

A sampled selection is **not re-runnable**: `random_n` renders `ORDER BY RANDOM()
LIMIT n` and draws a different sample every call, and `first_n` without an
`ORDER BY` is unordered on Postgres/MySQL. Re-issuing the selection compared the
baseline against unrelated rows — nearly every staged key looked deleted, so
`has_conflict` was permanently true and a sampled import could never be written
back. Those modes re-read exactly the staged keys instead
(`StagingManager.staged_keys` → `StreamingImporter.stream_by_keys`), which makes
changed/deleted exact at the cost of not seeing rows *added* to the source since
import. Re-import to pick those up.

Write-back re-baselines through the same path, so the rows it just wrote do not
read as third-party drift on the next check.

Detection needs primary-key columns. Without them every row falls back to a
synthetic `{"__rowid": n}` key, so the report comes back empty with an
explanatory note and write-back refuses to run.

## HTTP API (all under `/api/db`)

`state` · vault `unlock`/`lock`/`change-password` · `profiles` (GET masked / POST /
DELETE) · `test-connection` · `tables` · `columns` · `import` (SSE) ·
`session/<id>/rows` · `session/<id>/cell` (PUT) · `session/<id>/columns` (POST) ·
`session/<id>/process` (SSE) · `session/<id>/dry-run` · `session/<id>/check-conflicts`
· `session/<id>/writeback` (SSE) · `session/<id>/audit`.

## Offline / air-gapped install

All runtime dependencies are pure-Python or ship binary wheels, so they can be
vendored. On a machine with internet:

```bash
pip download -r requirements.txt -d wheelhouse
```

Copy `wheelhouse/` to the target machine and install without hitting the network:

```bash
pip install --no-index --find-links wheelhouse -r requirements.txt
```

Core DB deps: `SQLAlchemy`, `duckdb`, `cryptography`, `psycopg[binary]` (Postgres),
`PyMySQL` (MySQL/MariaDB). SQLite needs nothing extra. Optional engines add
`pymongo` (Mongo), `pyodbc` (SQL Server — also needs an OS ODBC driver), `oracledb`
(Oracle, thin mode), `sshtunnel`+`paramiko` (SSH tunnels), `azure-identity` (Azure
AD/IAM). Uncomment those in `requirements.txt` when enabling the corresponding
engine, and re-run the `pip download` step so their wheels are vendored too.

## Security notes

* Connection credentials are encrypted at rest (AES-256-GCM); the key is derived
  from the master password with scrypt and never written to disk. Lock the vault to
  wipe it from memory.
* Profiles are masked (`has_password` flags only) before they leave the server.
* SQL WHERE / ORDER BY fragments are passed through verbatim — this is a local,
  single-user tool pointed at the user's own database, so raw SQL is trusted by
  design. Do not expose these endpoints to untrusted clients. That licence is
  deliberate and narrow: it covers the fragments the user authored. Everything
  else that reaches SQL is bound as a parameter or validated — write-back values
  and keys are bound, identifiers go through the dialect preparer, and a staging
  column's DDL type is checked against `types_map.sanitize_ddl_type`.
* Write-back never claims more than it did. An UPDATE that matches no source row
  is audited as `nomatch`, left out of the applied counts, and NOT retired from
  staging, so the pending edit survives instead of being silently dropped.
