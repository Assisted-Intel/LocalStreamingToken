#!/usr/bin/env python3
"""
Connection manager. Builds a SQLAlchemy 2.x Engine (or a MongoDB client) from a
``ConnectionProfile`` and knows how to test a connection and introspect tables/
columns. SSL/TLS, SSH tunnels and Azure AD auth are layered in later phases but
the public surface is fixed here.

SQLAlchemy / drivers are imported lazily inside methods so the app boots without
them installed.

Phase status: SQLite/Postgres/MySQL implemented in Phase 2; Mongo in Phase 9;
SSH tunnel + Azure AD in Phase 10.
"""

from __future__ import annotations

from typing import Optional

from .models import ConnectionProfile, Engine

# SQLAlchemy driver URL prefixes per engine.
_DRIVER = {
    Engine.SQLITE.value: "sqlite",
    Engine.POSTGRES.value: "postgresql+psycopg",
    Engine.MYSQL.value: "mysql+pymysql",
    Engine.SQLSERVER.value: "mssql+pyodbc",
    Engine.ORACLE.value: "oracle+oracledb",
}

# Default ports so a user can leave the field blank.
_DEFAULT_PORT = {
    Engine.POSTGRES.value: 5432,
    Engine.MYSQL.value: 3306,
    Engine.SQLSERVER.value: 1433,
    Engine.ORACLE.value: 1521,
}


class ConnectionError_(Exception):
    """Raised when a connection cannot be established (wrapped, friendly message)."""


def build_url(profile: ConnectionProfile):
    """Compose a SQLAlchemy URL object from a profile's discrete fields, or wrap
    ``profile.url`` verbatim when the user supplied a full connection string.
    Returns a SQLAlchemy ``URL`` (or a str for a raw url)."""
    from sqlalchemy import URL, make_url

    if (profile.url or "").strip():
        return make_url(profile.url.strip())

    engine = profile.engine
    driver = _DRIVER.get(engine)
    if not driver:
        raise ConnectionError_(f"Unsupported engine {engine!r}.")

    if engine == Engine.SQLITE.value:
        # database holds the file path. Empty -> in-memory (rarely useful here).
        return URL.create("sqlite", database=profile.database or None)

    port = profile.port or _DEFAULT_PORT.get(engine)
    query = {}
    if engine == Engine.ORACLE.value and profile.extra.get("service_name"):
        query["service_name"] = profile.extra["service_name"]
    return URL.create(
        driver,
        username=profile.username or None,
        password=profile.password or None,
        host=profile.host or None,
        port=port,
        database=profile.database or None,
        query=query or {},
    )


def _ssl_connect_args(profile: ConnectionProfile) -> dict:
    """Driver-specific SSL/TLS connect args from the profile's cert paths."""
    if not profile.ssl_enabled:
        return {}
    e = profile.engine
    if e == Engine.POSTGRES.value:
        args = {"sslmode": profile.extra.get("sslmode", "require")}
        if profile.ssl_ca:
            args["sslrootcert"] = profile.ssl_ca
        if profile.ssl_cert:
            args["sslcert"] = profile.ssl_cert
        if profile.ssl_key:
            args["sslkey"] = profile.ssl_key
        return args
    if e == Engine.MYSQL.value:
        ssl = {}
        if profile.ssl_ca:
            ssl["ca"] = profile.ssl_ca
        if profile.ssl_cert:
            ssl["cert"] = profile.ssl_cert
        if profile.ssl_key:
            ssl["key"] = profile.ssl_key
        return {"ssl": ssl or {"ssl": True}}
    return {}


class ConnectionManager:
    """Stateless-ish factory; opens short-lived engines per operation. For Mongo
    it returns a pymongo client instead of a SQLAlchemy engine."""

    def build_engine(self, profile: ConnectionProfile, *, readonly: bool = True):
        """Return a SQLAlchemy Engine configured with SSL/connect args. ``readonly``
        opens the source read-only where the driver supports it (SQLite); for other
        engines the import path only ever issues SELECTs and never commits."""
        from sqlalchemy import create_engine

        if profile.engine == Engine.MONGODB.value:
            raise ConnectionError_("MongoDB uses mongo_client(), not build_engine().")

        url = build_url(profile)
        connect_args = dict(_ssl_connect_args(profile))
        connect_args.update(profile.extra.get("connect_args", {}) or {})

        try:
            eng = create_engine(url, connect_args=connect_args, pool_pre_ping=True,
                                future=True)
        except Exception as e:  # pragma: no cover - defensive
            raise ConnectionError_(str(e))

        if readonly and profile.engine == Engine.SQLITE.value:
            # Enforce read-only per-connection (path-agnostic, cross-platform). Any
            # write then fails with SQLITE_READONLY, so imports cannot mutate the source.
            from sqlalchemy import event

            @event.listens_for(eng, "connect")
            def _sqlite_readonly(dbapi_conn, _rec):  # noqa: ANN001
                dbapi_conn.execute("PRAGMA query_only=ON")

        return eng

    def mongo_client(self, profile: ConnectionProfile):
        """Return a pymongo client for a MongoDB profile. Phase 9."""
        raise NotImplementedError  # Phase 9

    def test_connection(self, profile: ConnectionProfile) -> dict:
        """Attempt to connect and probe. Returns {ok, engine, server_version,
        message}. Never raises — errors are captured into the dict."""
        from sqlalchemy import text

        if profile.engine == Engine.MONGODB.value:
            return {"ok": False, "engine": profile.engine,
                    "message": "MongoDB support arrives in Phase 9."}
        try:
            eng = self.build_engine(profile, readonly=True)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
                version = self._server_version(conn, profile.engine)
            eng.dispose()
            return {"ok": True, "engine": profile.engine, "server_version": version,
                    "message": f"Connected{f' — {version}' if version else ''}."}
        except Exception as e:
            return {"ok": False, "engine": profile.engine, "message": _friendly(e)}

    @staticmethod
    def _server_version(conn, engine: str) -> str:
        from sqlalchemy import text
        try:
            if engine == Engine.SQLITE.value:
                return "SQLite " + conn.execute(text("select sqlite_version()")).scalar_one()
            if engine == Engine.POSTGRES.value:
                return conn.execute(text("show server_version")).scalar_one()
            if engine == Engine.MYSQL.value:
                return "MySQL " + conn.execute(text("select version()")).scalar_one()
        except Exception:
            pass
        return ""

    def list_tables(self, profile: ConnectionProfile) -> list:
        """Return table + view names available on the profile."""
        from sqlalchemy import inspect
        eng = self.build_engine(profile, readonly=True)
        try:
            insp = inspect(eng)
            names = list(insp.get_table_names())
            try:
                names += list(insp.get_view_names())
            except Exception:
                pass
            return sorted(set(names))
        finally:
            eng.dispose()

    def list_columns(self, profile: ConnectionProfile, table: str) -> list:
        """Return [{name, type, nullable, primary_key}] for a table via inspect."""
        from sqlalchemy import inspect
        eng = self.build_engine(profile, readonly=True)
        try:
            insp = inspect(eng)
            try:
                pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            except Exception:
                pk = set()
            cols = []
            for c in insp.get_columns(table):
                cols.append({
                    "name": c["name"],
                    "type": str(c.get("type")),
                    "nullable": bool(c.get("nullable", True)),
                    "primary_key": c["name"] in pk,
                })
            return cols
        finally:
            eng.dispose()


def _friendly(exc: Exception) -> str:
    """Trim a driver exception to something a user can act on."""
    msg = str(exc)
    # SQLAlchemy wraps driver errors with a "(Background on this error...)" tail.
    msg = msg.split("(Background on this error", 1)[0].strip()
    return msg[:400] or exc.__class__.__name__
