#!/usr/bin/env python3
"""End-to-end tests for the Database tab's routes (/api/db/*).

These drive the real Flask app against a throwaway data+settings tree and a real
SQLite source file. No network, no Ollama.

The unit-level guarantees live in test_writeback.py; what only shows up here is the
wiring between the route generators and the engines underneath them:

* the whole loop — unlock, save a connection, import, edit, dry-run, check
  conflicts, write back, check again — has to leave the session CLEAN, because the
  route (not the engine) is what retires the applied cells and re-baselines them;
* that loop has to work for a SAMPLED import, whose selection cannot be re-run;
* a locked vault has to answer 423 rather than quietly connecting with no password;
* the grid has to be able to write NULL into a typed column.
"""

import json
import sqlite3

import pytest

from conftest import make_client


# --------------------------- harness ---------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """The shared client (tests/conftest.py), wrapped to release any staging file a
    previous test still holds — this suite is the only one that opens them."""
    from app.database import staging as db_staging
    db_staging.close_all()
    yield make_client(tmp_path, monkeypatch)
    db_staging.close_all()


@pytest.fixture
def source(tmp_path):
    """A SQLite source table with a primary key and 20 rows."""
    path = tmp_path / "source.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, "
                "note TEXT, score INTEGER)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?, ?)",
                    [(i, f"person{i}", "", i * 10) for i in range(1, 21)])
    con.commit()
    con.close()
    return path


def sse_frames(resp):
    """Parse an SSE response body into [(event, data-dict), ...]."""
    out = []
    for block in resp.get_data(as_text=True).split("\n\n"):
        event, data = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event and data:
            out.append((event, json.loads(data)))
    return out


def first(frames, name):
    return next(d for e, d in frames if e == name)


def setup_session(client, source, mode="full", n=100):
    """Unlock the vault, save a connection, import, and return the session id."""
    assert client.post("/api/db/vault/unlock", json={"password": "vault-pw"}).status_code == 200
    r = client.post("/api/db/profiles", json={"profile": {
        "name": "src", "engine": "sqlite", "database": str(source)}})
    assert r.status_code == 200, r.get_data(as_text=True)
    pid = r.get_json()["profile"]["id"]

    r = client.post("/api/db/import", json={
        "profile_id": pid, "table": "people", "name": "T",
        "selection": {"mode": mode, "n": n}})
    assert r.status_code == 200, r.get_data(as_text=True)
    done = first(sse_frames(r), "done")
    return done["session_id"], done["row_count"], pid


def rows_of(client, sid):
    r = client.get(f"/api/db/session/{sid}/rows?limit=500")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def written(source):
    con = sqlite3.connect(source)
    n = con.execute("SELECT COUNT(*) FROM people WHERE note LIKE 'edited-%'").fetchone()[0]
    con.close()
    return n


# --------------------------- the round trip ---------------------------

@pytest.mark.parametrize("mode,n", [("full", 100), ("random_n", 5), ("first_n", 5)])
def test_the_whole_loop_leaves_the_session_clean(client, source, mode, n):
    """Import -> edit -> dry-run -> check -> write back -> check again.

    The second check is the point. Write-back is what re-baselines the staging table,
    and the ROUTE does that (the engine only reports which cells committed). If it
    does not, the session's own writes read as third-party drift and the default
    on_conflict='abort' locks it out of ever writing again. For a sampled selection
    this used to fail even before writing: re-running `random_n` drew a fresh sample,
    so every staged key looked deleted.
    """
    sid, count, _ = setup_session(client, source, mode=mode, n=n)
    assert count == (20 if mode == "full" else n)

    page = rows_of(client, sid)
    assert page["total"] == count
    assert page["column_types"]["score"].upper().startswith(("INT", "BIGINT"))

    for row in page["rows"]:
        r = client.put(f"/api/db/session/{sid}/cell", json={
            "rowid": row["__rowid"], "column": "note", "value": f"edited-{row['__rowid']}"})
        assert r.status_code == 200, r.get_data(as_text=True)
    assert rows_of(client, sid)["dirty"] == count

    dry = client.post(f"/api/db/session/{sid}/dry-run", json={}).get_json()
    assert dry["total_statements"] == count
    assert dry["skipped_columns"] == []

    pre = client.post(f"/api/db/session/{sid}/check-conflicts", json={}).get_json()
    assert pre["has_conflict"] is False, pre

    frames = sse_frames(client.post(f"/api/db/session/{sid}/writeback",
                                    json={"approved": True, "mode": "bulk"}))
    done = first(frames, "done")
    assert done["applied_rows"] == count
    assert not [d for e, d in frames if e == "warn"], "no warnings expected on a clean run"
    assert written(source) == count

    post = client.post(f"/api/db/session/{sid}/check-conflicts", json={}).get_json()
    assert post["has_conflict"] is False, post
    assert rows_of(client, sid)["dirty"] == 0, "applied cells must be retired"


def test_a_second_write_back_after_a_clean_one_has_nothing_to_do(client, source):
    """Proof the session is genuinely reusable rather than merely not erroring."""
    sid, count, _ = setup_session(client, source)
    for row in rows_of(client, sid)["rows"]:
        client.put(f"/api/db/session/{sid}/cell", json={
            "rowid": row["__rowid"], "column": "note", "value": f"edited-{row['__rowid']}"})
    first(sse_frames(client.post(f"/api/db/session/{sid}/writeback",
                                 json={"approved": True})), "done")

    again = first(sse_frames(client.post(f"/api/db/session/{sid}/writeback",
                                         json={"approved": True})), "done")
    assert again["applied_rows"] == 0
    assert "Nothing to write" in again.get("message", "")


def test_a_real_external_change_still_aborts_the_write(client, source):
    sid, _, _ = setup_session(client, source)
    for row in rows_of(client, sid)["rows"][:3]:
        client.put(f"/api/db/session/{sid}/cell", json={
            "rowid": row["__rowid"], "column": "note", "value": f"edited-{row['__rowid']}"})

    con = sqlite3.connect(source)
    con.execute("UPDATE people SET name = 'someone-else' WHERE id = 1")
    con.commit()
    con.close()

    report = client.post(f"/api/db/session/{sid}/check-conflicts", json={}).get_json()
    assert report["has_conflict"] and len(report["changed"]) == 1

    frames = sse_frames(client.post(f"/api/db/session/{sid}/writeback",
                                    json={"approved": True}))       # on_conflict=abort
    assert [e for e, _ in frames][-1] == "error"
    assert written(source) == 0


def test_write_back_without_approval_writes_nothing(client, source):
    sid, _, _ = setup_session(client, source)
    for row in rows_of(client, sid)["rows"][:3]:
        client.put(f"/api/db/session/{sid}/cell", json={
            "rowid": row["__rowid"], "column": "note", "value": f"edited-{row['__rowid']}"})
    frames = sse_frames(client.post(f"/api/db/session/{sid}/writeback", json={}))
    assert first(frames, "error")["message"] == "Write-back not approved."
    assert written(source) == 0


# --------------------------- editing ---------------------------

def test_a_typed_column_can_be_set_to_null(client, source):
    """The grid sends null for an emptied non-text cell. Sending "" made DuckDB fail
    the cast, so a numeric cell could never be cleared."""
    sid, _, _ = setup_session(client, source)
    r = client.put(f"/api/db/session/{sid}/cell",
                   json={"rowid": 0, "column": "score", "value": None})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert rows_of(client, sid)["rows"][0]["score"] is None


def test_an_uncastable_value_is_a_400_not_a_500(client, source):
    sid, _, _ = setup_session(client, source)
    r = client.put(f"/api/db/session/{sid}/cell",
                   json={"rowid": 0, "column": "score", "value": "not-a-number"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_an_added_column_is_staging_only_and_reported_as_skipped(client, source):
    sid, _, _ = setup_session(client, source)
    r = client.post(f"/api/db/session/{sid}/columns",
                    json={"name": "Summary", "ctype": "output"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert "Summary" in rows_of(client, sid)["columns"]

    client.put(f"/api/db/session/{sid}/cell",
               json={"rowid": 0, "column": "Summary", "value": "ai text"})
    dry = client.post(f"/api/db/session/{sid}/dry-run", json={}).get_json()
    # The source has no Summary column, so it must be reported, never invented.
    assert dry["skipped_columns"] == ["Summary"]
    assert dry["total_statements"] == 0


def test_a_client_supplied_column_type_cannot_reach_the_ddl(client, source):
    """The type lands in `ALTER TABLE ... ADD COLUMN`, which cannot be parameterized."""
    sid, _, _ = setup_session(client, source)
    r = client.post(f"/api/db/session/{sid}/columns", json={
        "name": "X", "duckdb_type": "VARCHAR; DROP TABLE staged;--"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert "X" in rows_of(client, sid)["columns"]           # table intact
    assert rows_of(client, sid)["column_types"]["X"].upper() == "VARCHAR"


# --------------------------- the vault gate ---------------------------

def test_a_locked_vault_answers_423_rather_than_connecting_blind(client, source):
    """VaultLocked subclasses VaultError, and _resolve_profile's broad catch used to
    swallow it — so a saved profile connected with a BLANK password and the user got
    an authentication failure from their own database instead of "unlock the vault"."""
    client.post("/api/db/vault/unlock", json={"password": "vault-pw"})
    pid = client.post("/api/db/profiles", json={"profile": {
        "name": "src", "engine": "sqlite", "database": str(source),
        "username": "u", "password": "p"}}).get_json()["profile"]["id"]
    client.post("/api/db/vault/lock", json={})

    for path, body in (("/api/db/tables", {"profile_id": pid}),
                       ("/api/db/columns", {"profile_id": pid, "table": "people"}),
                       ("/api/db/test-connection", {"profile": {"id": pid, "engine": "sqlite"}})):
        r = client.post(path, json=body)
        assert r.status_code == 423, f"{path} -> {r.status_code}"
        assert r.get_json()["locked"] is True


def test_a_wrong_master_password_is_a_400_not_a_500(client):
    client.post("/api/db/vault/unlock", json={"password": "vault-pw"})
    client.post("/api/db/vault/lock", json={})
    r = client.post("/api/db/vault/unlock", json={"password": "nope"})
    assert r.status_code == 400
    assert "password" in r.get_json()["error"].lower()


def test_saved_passwords_never_leave_the_server(client, source):
    client.post("/api/db/vault/unlock", json={"password": "vault-pw"})
    client.post("/api/db/profiles", json={"profile": {
        "name": "src", "engine": "postgresql", "host": "h",
        "database": "d", "username": "u", "password": "s3cret"}})
    body = client.get("/api/db/profiles").get_data(as_text=True)
    assert "s3cret" not in body
    assert json.loads(body)["profiles"][0]["has_password"] is True


# --------------------------- input handling ---------------------------

def test_garbage_paging_args_do_not_500(client, source):
    sid, _, _ = setup_session(client, source)
    r = client.get(f"/api/db/session/{sid}/rows?offset=abc&limit=-9")
    assert r.status_code == 200
    assert r.get_json()["rows"], "should fall back to sane defaults"
    assert client.get(f"/api/db/session/{sid}/audit?limit=xyz").status_code == 200


def test_a_missing_session_is_a_404(client):
    assert client.get("/api/db/session/nope/rows").status_code == 404
    assert client.post("/api/db/session/nope/dry-run", json={}).status_code == 404


def test_deleting_a_session_drops_its_staging_file_only(client, source):
    from app.database.staging import staging_path

    sid, _, _ = setup_session(client, source)
    assert staging_path(sid).exists()
    assert client.delete(f"/api/db/session/{sid}").status_code == 200
    assert not staging_path(sid).exists()

    con = sqlite3.connect(source)          # the SOURCE is untouched
    assert con.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 20
    con.close()
