#!/usr/bin/env python3
"""
Shared, dependency-free data models for the Database Processing feature.

These are plain dataclasses/enums with no third-party imports, so they are safe to
import anywhere (routes, tests) regardless of whether the DB drivers are installed.
Everything that crosses the wire to the browser is a plain dict — use ``to_dict``/
``from_dict`` helpers; secrets are stripped by the vault layer, not here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


def _now() -> str:
    return datetime.utcnow().isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class Engine(str, Enum):
    """Supported source engines. Phase 1-3 target the first three; the rest are
    wired in later phases but declared here so the profile shape is stable."""
    SQLITE = "sqlite"
    POSTGRES = "postgresql"
    MYSQL = "mysql"          # MySQL / MariaDB
    SQLSERVER = "mssql"      # Phase 10
    ORACLE = "oracle"        # Phase 10
    MONGODB = "mongodb"      # Phase 9


class ColumnType(str, Enum):
    """A staging column's role during AI processing (POI's model minus image/
    foursquare). ``source`` columns come straight from the imported table."""
    SOURCE = "source"        # imported verbatim from the source DB
    WEB_SOURCE = "web_source"  # populated by core.web_search(query) before prompts run
    PROMPT = "prompt"        # filled by an LLM from a {Column} template
    OUTPUT = "output"        # alias of prompt kept for clarity in the UI


class SelectionMode(str, Enum):
    FIRST_N = "first_n"
    RANDOM_N = "random_n"
    FULL = "full"
    CUSTOM = "custom"        # SQL WHERE clause / Mongo filter


class WriteMode(str, Enum):
    BULK = "bulk"            # batched executemany / bulk_write (preferred)
    ROW = "row"             # one statement per row, individually audited


class ConflictKind(str, Enum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"       # row exists in source but its fingerprint differs
    NEW = "new"               # row appeared in source since import
    DELETED = "deleted"       # row vanished from source since import


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ConnectionProfile:
    """A saved connection. Lives ENCRYPTED in the vault — credential fields here
    are only ever in memory (after unlock) or inside the AES-GCM blob at rest."""
    id: str = field(default_factory=new_id)
    name: str = ""
    engine: str = Engine.SQLITE.value
    # Either a full SQLAlchemy/Mongo URL (takes precedence) OR discrete fields.
    url: str = ""
    host: str = ""
    port: Optional[int] = None
    database: str = ""       # db name, or file path for SQLite
    username: str = ""
    password: str = ""       # secret — masked before leaving the server
    # TLS / SSL.
    ssl_enabled: bool = False
    ssl_ca: str = ""         # path to CA cert
    ssl_cert: str = ""       # path to client cert
    ssl_key: str = ""        # path to client key
    # Modern auth (Phase 10).
    ssh_tunnel: Optional[dict] = None    # {host, port, user, key_path/password, remote_host, remote_port}
    azure_auth: bool = False             # use azure-identity DefaultAzureCredential for the token
    extra: dict = field(default_factory=dict)  # driver-specific connect args
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    SECRET_FIELDS = ("password", "ssl_key")

    def to_dict(self) -> dict:
        return asdict(self)

    def masked(self) -> dict:
        """Safe to send to the browser: secret fields replaced by has_<field>."""
        d = self.to_dict()
        for k in self.SECRET_FIELDS:
            d[f"has_{k}"] = bool(d.pop(k, None))
        if d.get("ssh_tunnel"):
            t = dict(d["ssh_tunnel"])
            t["has_password"] = bool(t.pop("password", None))
            t["has_key"] = bool(t.pop("key_path", None) or t.pop("private_key", None))
            d["ssh_tunnel"] = t
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConnectionProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class ColumnDef:
    """One staging column and (for non-source columns) how to fill it."""
    name: str = ""
    ctype: str = ColumnType.SOURCE.value
    source_type: str = ""     # original DB type string (best-effort preserved)
    duckdb_type: str = "VARCHAR"
    # For WEB_SOURCE / PROMPT columns:
    prompt_template: str = ""       # {Column} placeholders (evals.fill_prompt syntax)
    input_columns: list = field(default_factory=list)
    search_query: str = ""          # web-source: {Column} template → core.web_search
    domains: list = field(default_factory=list)  # optional allow-list for web search

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectionSpec:
    """How many / which rows to import."""
    mode: str = SelectionMode.FIRST_N.value
    n: int = 100
    where: str = ""          # custom SQL WHERE / Mongo filter (JSON) when mode == custom
    order_by: str = ""       # optional stable ordering (recommended for first_n)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImportSession:
    """Non-secret record of one import. Persisted via Store.db_projects; the actual
    rows live in data/db/staging/staging_<id>.duckdb."""
    id: str = field(default_factory=new_id)
    name: str = "Untitled import"
    profile_id: str = ""
    table: str = ""          # table or collection name
    selection: dict = field(default_factory=lambda: SelectionSpec().to_dict())
    columns: list = field(default_factory=list)   # [ColumnDef.to_dict()]
    key_columns: list = field(default_factory=list)  # source PK(s) for write-back matching
    row_count: int = 0
    status: str = "new"       # new | importing | staged | processing | ready | writing | done | error
    staging_table: str = "staged"
    source_fingerprint: dict = field(default_factory=dict)  # for conflict detection
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImportSession":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class DryRunResult:
    statements: list = field(default_factory=list)  # [{sql, params, literal, key}]
    total_rows: int = 0
    total_statements: int = 0
    truncated: bool = False
    dialect: str = ""
    skipped_columns: list = field(default_factory=list)  # staging-only cols absent from source
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConflictReport:
    changed: list = field(default_factory=list)   # keys changed in source since import
    new: list = field(default_factory=list)       # keys new in source
    deleted: list = field(default_factory=list)   # keys removed from source
    unchanged: int = 0
    has_conflict: bool = False
    note: str = ""                                # e.g. why detection was limited

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditEntry:
    ts: str = field(default_factory=_now)
    session_id: str = ""
    table: str = ""
    key: dict = field(default_factory=dict)   # {key_col: value}
    column: str = ""
    old: Any = None
    new: Any = None
    statement: str = ""
    mode: str = WriteMode.BULK.value
    status: str = "ok"        # ok | error
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
