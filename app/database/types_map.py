#!/usr/bin/env python3
"""
Type preservation: map a source column's SQLAlchemy type to the closest DuckDB
column type when creating a staging table, so integers stay integers, decimals
keep precision, dates stay dates, etc. Anything we don't recognise falls back to
VARCHAR (lossless for round-tripping text) rather than guessing wrong.

Pure/dependency-light: it inspects the *class name* of a SQLAlchemy type object,
so it does not require the type instance to come from any particular dialect and
imports nothing heavy at module load.
"""

from __future__ import annotations

import re

# SQLAlchemy generic type class-name  ->  DuckDB type.
# We match on the uppercased class name (e.g. "INTEGER", "BIGINT", "NUMERIC").
_MAP = {
    "SMALLINT": "SMALLINT", "INTEGER": "INTEGER", "INT": "INTEGER", "BIGINT": "BIGINT",
    "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN",
    "FLOAT": "DOUBLE", "REAL": "REAL", "DOUBLE": "DOUBLE", "DOUBLE_PRECISION": "DOUBLE",
    "NUMERIC": "DECIMAL", "DECIMAL": "DECIMAL",
    "DATE": "DATE", "TIME": "TIME",
    "DATETIME": "TIMESTAMP", "TIMESTAMP": "TIMESTAMP",
    "UUID": "UUID",
    "LARGEBINARY": "BLOB", "BINARY": "BLOB", "VARBINARY": "BLOB", "BLOB": "BLOB",
    "JSON": "JSON", "JSONB": "JSON",
    "TEXT": "VARCHAR", "CLOB": "VARCHAR", "STRING": "VARCHAR",
    "VARCHAR": "VARCHAR", "NVARCHAR": "VARCHAR", "CHAR": "VARCHAR", "NCHAR": "VARCHAR",
}


# DuckDB type names a client is allowed to name when adding a staging column. The
# type goes into DDL (`ALTER TABLE ... ADD COLUMN "x" <type>`), which cannot be
# parameterized, so it is validated against this list rather than interpolated as
# given. Unlike the WHERE/ORDER BY fragments — which the feature documents as
# deliberately trusted, user-authored SQL — nothing in the UI ever sends anything
# but VARCHAR here, so there is no expressiveness to preserve.
_ALLOWED_DDL = {
    "BOOLEAN", "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC",
    "VARCHAR", "TEXT", "STRING", "BLOB", "BIT", "UUID", "JSON",
    "DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ", "INTERVAL",
}
# DECIMAL(p,s) / VARCHAR(n) — a bare width/precision suffix on an allowed base type.
_PARAM_DDL = re.compile(r"^([A-Z_]+)\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\)$")


def sanitize_ddl_type(raw: str, default: str = "VARCHAR") -> str:
    """Return ``raw`` if it names an allowed DuckDB type, else ``default``.

    Accepts a bare name (``BIGINT``) or a parameterized one (``DECIMAL(18,2)``).
    Case-insensitive; always returns the canonical upper-case form."""
    name = (raw or "").strip().upper()
    if not name:
        return default
    if name in _ALLOWED_DDL:
        return name
    m = _PARAM_DDL.match(name)
    if m and m.group(1) in _ALLOWED_DDL:
        return name
    return default


def duckdb_type_for(sa_type) -> str:
    """Return the DuckDB type string for a SQLAlchemy type object (or a raw type
    string). Preserves DECIMAL(precision, scale) when available."""
    if sa_type is None:
        return "VARCHAR"
    name = type(sa_type).__name__.upper() if not isinstance(sa_type, str) else sa_type.upper()

    # Preserve decimal precision/scale where the type object exposes it.
    if name in ("NUMERIC", "DECIMAL"):
        prec = getattr(sa_type, "precision", None)
        scale = getattr(sa_type, "scale", None)
        if prec:
            return f"DECIMAL({prec},{scale or 0})"
        return "DECIMAL(38,9)"

    if name in _MAP:
        return _MAP[name]

    # Best-effort substring match for dialect-specific class names
    # (e.g. Postgres "TIMESTAMP" subclasses, MySQL "TINYINT").
    for key, ddl in _MAP.items():
        if key in name:
            return ddl
    if "INT" in name:
        return "BIGINT"
    if "CHAR" in name or "TEXT" in name:
        return "VARCHAR"
    return "VARCHAR"
