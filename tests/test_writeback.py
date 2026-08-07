#!/usr/bin/env python3
"""Tests for the Database tab's write-back path (app/database/).

Write-back is the only place the app opens a user's real database writable, so the
guarantees under test are about honesty and repeatability rather than features:

  * a cancelled BULK run rolls back AND says so — it must not audit or count writes
    the database never kept (the audit log is the record of what touched real data);
  * approval is required in the request, not inheritable from stored session state;
  * after a successful write-back the session is clean and its conflict baseline
    reflects what we just wrote, so the user's own changes don't come back as
    third-party drift and lock the session out of ever writing again;
  * a genuine external change is still detected;
  * a run that dies partway still reports which rows committed, so the caller can
    retire exactly those;
  * an UPDATE that matches nothing is not counted, audited, or retired as a write;
  * conflict detection is stable for a SAMPLED selection, which cannot be re-run.

A real SQLite file is the source and a real DuckDB file is the staging store — the
engine's transaction semantics are the point, so mocking them would test nothing.
"""

import json
import sqlite3
import threading

import pytest

from app import core
from app.database.audit import AuditLog
from app.database.conflict import ConflictDetector
from app.database.connections import ConnectionManager
from app.database.importer import StreamingImporter
from app.database.models import ConnectionProfile, SelectionSpec
from app.database.staging import StagingManager
from app.database.writeback import WriteBackEngine

COLUMNS = [{"name": "id", "ctype": "source", "duckdb_type": "BIGINT"},
           {"name": "name", "ctype": "source", "duckdb_type": "VARCHAR"},
           {"name": "note", "ctype": "source", "duckdb_type": "VARCHAR"}]
ROWS = 20


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A SQLite source imported into a DuckDB staging table, with every row edited."""
    monkeypatch.setattr(core, "DB_STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(core, "DB_AUDIT_DIR", tmp_path / "audit")
    core.DB_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    core.DB_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    src = tmp_path / "source.sqlite"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, note TEXT)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?)",
                    [(i, f"person{i}", "") for i in range(1, ROWS + 1)])
    con.commit()
    con.close()

    profile = ConnectionProfile(name="t", engine="sqlite", database=str(src))
    # "full" is the real SelectionMode. This used to say "all", which is not a member
    # and silently fell through to the FULL branch — so every test here exercised the
    # full-table path while appearing to be mode-agnostic.
    session = {"id": "sess", "table": "people", "staging_table": "staged",
               "key_columns": ["id"], "columns": COLUMNS, "selection": {"mode": "full"}}

    conns = ConnectionManager()
    sm = StagingManager("sess")
    sm.create_table(COLUMNS)
    for chunk in StreamingImporter(conns).stream(profile, "people", SelectionSpec(mode="full")):
        sm.insert_chunk(chunk, COLUMNS, ["id"])
    for rid in range(ROWS):
        sm.update_cell(rid, "note", f"enriched-{rid}")

    yield {"src": src, "profile": profile, "session": session, "conns": conns, "sm": sm}
    sm.close()


def run(bench, **kw):
    """Drive the engine to completion and return its final `done` frame."""
    kw.setdefault("approved", True)
    kw.setdefault("on_conflict", "overwrite")
    kw.setdefault("batch_size", 5)
    engine = WriteBackEngine(bench["conns"])
    done = None
    for kind, data in engine.run(bench["profile"], bench["session"],
                                 bench["sm"].dirty_changes(), **kw):
        if kind == "done":
            done = data
    return done


def source_written(bench):
    con = sqlite3.connect(bench["src"])
    n = con.execute("SELECT COUNT(*) FROM people WHERE note LIKE 'enriched-%'").fetchone()[0]
    con.close()
    return n


def test_write_back_applies_and_audits(bench):
    done = run(bench)
    assert done["applied_rows"] == ROWS
    assert source_written(bench) == ROWS
    assert AuditLog("sess").count() == ROWS


def test_cancelled_bulk_run_writes_nothing_and_claims_nothing(bench):
    """Stopped before the first batch: rollback, no audit, no counted rows."""
    stop = threading.Event()
    stop.set()
    done = run(bench, mode="bulk", stop_event=stop)
    assert done["stopped"] is True
    assert done["applied_rows"] == 0
    assert source_written(bench) == 0
    assert AuditLog("sess").count() == 0


def test_stop_midway_through_the_transaction_rolls_the_whole_thing_back(bench):
    """The counters and the audit log used to be written from INSIDE the transaction,
    so a cancel reported rows as applied that the rollback then discarded."""
    stop = threading.Event()
    engine = WriteBackEngine(bench["conns"])
    done = None
    for kind, data in engine.run(bench["profile"], bench["session"],
                                 bench["sm"].dirty_changes(), approved=True,
                                 on_conflict="overwrite", batch_size=5, stop_event=stop):
        if kind == "progress":
            stop.set()          # cancel after a batch has already executed
        if kind == "done":
            done = data
    assert done["applied_rows"] == 0
    assert source_written(bench) == 0
    assert AuditLog("sess").count() == 0


def test_write_back_requires_approval_in_the_request(bench):
    engine = WriteBackEngine(bench["conns"])
    frames = list(engine.run(bench["profile"], bench["session"],
                             bench["sm"].dirty_changes(), approved=False))
    assert frames[0][0] == "error"
    assert source_written(bench) == 0


def test_retire_and_rebaseline_leave_the_session_reusable(bench):
    """Without this the session is a one-shot: it stays dirty forever, and its own
    writes read as someone else's changes on the next conflict check."""
    done = run(bench)
    det = ConflictDetector(bench["conns"])

    # Before re-baselining, our own writes look exactly like third-party drift.
    assert det.check(bench["profile"], bench["session"]).has_conflict

    bench["sm"].retire_changes(done["applied_pairs"])
    bench["sm"].rebaseline(det.current_fingerprints(bench["profile"], bench["session"]))

    assert bench["sm"].dirty_changes() == []
    assert not det.check(bench["profile"], bench["session"]).has_conflict


def test_a_real_external_change_is_still_detected_after_rebaseline(bench):
    """Re-baselining must not blunt conflict detection."""
    done = run(bench)
    det = ConflictDetector(bench["conns"])
    bench["sm"].retire_changes(done["applied_pairs"])
    bench["sm"].rebaseline(det.current_fingerprints(bench["profile"], bench["session"]))

    con = sqlite3.connect(bench["src"])
    con.execute("UPDATE people SET name = 'someone-else' WHERE id = 3")
    con.commit()
    con.close()

    report = det.check(bench["profile"], bench["session"])
    assert report.has_conflict and len(report.changed) == 1


def test_abort_on_conflict_writes_nothing(bench):
    con = sqlite3.connect(bench["src"])
    con.execute("UPDATE people SET name = 'someone-else' WHERE id = 3")
    con.commit()
    con.close()
    done = run(bench, on_conflict="abort")
    assert done is None                 # aborted before any `done` frame
    assert source_written(bench) == 0


# --------------------------- row-by-row mode ---------------------------

def frames(bench, **kw):
    """Every frame the engine emits, so the failure paths can be inspected."""
    kw.setdefault("approved", True)
    kw.setdefault("on_conflict", "overwrite")
    engine = WriteBackEngine(bench["conns"])
    return list(engine.run(bench["profile"], bench["session"],
                           bench["sm"].dirty_changes(), **kw))


def test_row_mode_reports_what_committed_before_an_error(bench):
    """A row-mode run that aborts on error must still emit `done`.

    Each row commits in its own transaction, so by the time one fails the earlier
    ones are already in the user's database. Returning early skipped the `done`
    frame, so the route never learned which cells to retire — the session stayed
    dirty, and on the next conflict check its OWN writes came back as third-party
    drift, which under the default on_conflict='abort' locked it out of ever
    writing again.
    """
    # Break row id=5 by making its column unwritable, so exactly one UPDATE fails.
    con = sqlite3.connect(bench["src"])
    con.execute("CREATE TRIGGER boom BEFORE UPDATE ON people WHEN NEW.id = 5 "
                "BEGIN SELECT RAISE(ABORT, 'nope'); END")
    con.commit()
    con.close()

    got = frames(bench, mode="row", continue_on_error=False)
    kinds = [k for k, _ in got]
    done = dict(got[-1][1])

    assert "done" in kinds, "the run must not end without a done frame"
    assert done["partial"] is True
    assert done["error"] == "" or "nope" in done["error"]
    # id=5 is the 5th staged row (rowid 4), so 4 rows committed before the failure.
    assert done["applied_rows"] == 4
    assert source_written(bench) == 4
    assert len(done["applied_pairs"]) == 4

    # The committed rows can now be retired and re-baselined -> session stays usable.
    det = ConflictDetector(bench["conns"])
    bench["sm"].retire_changes(done["applied_pairs"])
    bench["sm"].rebaseline(det.current_fingerprints(bench["profile"], bench["session"]))
    # The 16 unwritten rows are still pending; nothing was lost.
    assert {c["rowid"] for c in bench["sm"].dirty_changes()} == set(range(4, ROWS))
    assert not det.check(bench["profile"], bench["session"]).has_conflict


def test_continue_on_error_writes_every_other_row(bench):
    con = sqlite3.connect(bench["src"])
    con.execute("CREATE TRIGGER boom BEFORE UPDATE ON people WHEN NEW.id = 5 "
                "BEGIN SELECT RAISE(ABORT, 'nope'); END")
    con.commit()
    con.close()

    got = frames(bench, mode="row", continue_on_error=True)
    done = dict(got[-1][1])
    assert done["applied_rows"] == ROWS - 1
    assert source_written(bench) == ROWS - 1


def test_an_update_matching_no_source_row_is_not_counted_as_written(bench):
    """The statement succeeds and touches nothing. Counting it would retire the
    pending edit — silently losing it — and log a write that never happened."""
    con = sqlite3.connect(bench["src"])
    con.execute("DELETE FROM people WHERE id = 7")
    con.commit()
    con.close()

    got = frames(bench, mode="row", on_conflict="overwrite", continue_on_error=True)
    done = dict(got[-1][1])
    warns = [d for k, d in got if k == "warn"]

    assert done["applied_rows"] == ROWS - 1
    assert done["unmatched_rows"] == 1
    assert any("matched no source row" in (w.get("message") or "") for w in warns)
    # The vanished row's edit is NOT retired, so it survives as still-pending.
    bench["sm"].retire_changes(done["applied_pairs"])
    assert {c["rowid"] for c in bench["sm"].dirty_changes()} == {6}   # id 7 -> rowid 6
    # ...and it is audited honestly rather than as a successful write.
    entries = AuditLog("sess").read(limit=1000)
    assert [e["status"] for e in entries].count("nomatch") == 1


# ------------------- conflict detection on a SAMPLED import -------------------

SAMPLE_N = 5


@pytest.fixture
def sampled(bench):
    """A SECOND session over the same source, holding a random sample of its rows.

    A random_n selection renders `ORDER BY RANDOM() LIMIT n`, so it is not
    re-runnable — the detector must re-read the STAGED KEYS instead. The sample must
    be a strict subset (SAMPLE_N < ROWS) for that to be observable: with n >= the
    table size, `LIMIT n` returns everything and a re-sample looks identical.
    """
    sel = SelectionSpec(mode="random_n", n=SAMPLE_N)
    sm = StagingManager("samp")
    sm.create_table(COLUMNS)
    for chunk in StreamingImporter(bench["conns"]).stream(bench["profile"], "people", sel):
        sm.insert_chunk(chunk, COLUMNS, ["id"])
    assert sm.row_count() == SAMPLE_N
    for rid in range(SAMPLE_N):
        sm.update_cell(rid, "note", f"enriched-{rid}")

    session = dict(bench["session"], id="samp", selection=sel.to_dict())
    out = dict(bench, session=session, sm=sm)
    yield out
    sm.close()


def test_a_sampled_selection_is_not_re_sampled_when_checking_conflicts(sampled):
    """Re-issuing the selection drew a fresh random sample each call, so the
    baseline was diffed against unrelated rows: nearly every staged key looked
    deleted, has_conflict was permanently true, and — under the default
    on_conflict='abort' — a randomly-sampled import could never be written back."""
    det = ConflictDetector(sampled["conns"])
    first = det.current_fingerprints(sampled["profile"], sampled["session"])
    second = det.current_fingerprints(sampled["profile"], sampled["session"])

    assert first == second, "two reads of an unchanged source must agree"
    assert set(first) == set(sampled["sm"].fingerprints()), "must re-read the staged keys"

    report = det.check(sampled["profile"], sampled["session"])
    assert not report.has_conflict
    assert report.unchanged == SAMPLE_N
    assert report.deleted == [] and report.changed == []


def test_a_sampled_import_can_be_written_back_and_stays_clean(sampled):
    done = run(sampled, on_conflict="abort")     # the default, and the one that broke
    assert done["applied_rows"] == SAMPLE_N
    assert source_written(sampled) == SAMPLE_N

    det = ConflictDetector(sampled["conns"])
    sampled["sm"].retire_changes(done["applied_pairs"])
    sampled["sm"].rebaseline(det.current_fingerprints(sampled["profile"], sampled["session"]))
    assert not det.check(sampled["profile"], sampled["session"]).has_conflict


def test_a_sampled_check_still_catches_real_drift(sampled):
    """Re-reading only the staged keys must not blunt detection of real changes."""
    staged = [k["id"] for k in sampled["sm"].staged_keys()]   # which rows were sampled
    con = sqlite3.connect(sampled["src"])
    con.execute("UPDATE people SET name = 'someone-else' WHERE id = ?", (staged[0],))
    con.execute("DELETE FROM people WHERE id = ?", (staged[1],))
    con.commit()
    con.close()

    report = ConflictDetector(sampled["conns"]).check(sampled["profile"], sampled["session"])
    assert report.has_conflict
    assert len(report.changed) == 1 and len(report.deleted) == 1
    # A keyed re-read cannot see rows ADDED since import — say so rather than
    # reporting an empty list as though the table had been scanned.
    assert report.new == []
    assert "not visible here" in report.note


def test_a_sampled_check_does_not_report_new_source_rows_as_conflicts(sampled):
    con = sqlite3.connect(sampled["src"])
    con.execute("INSERT INTO people VALUES (999, 'newcomer', '')")
    con.commit()
    con.close()

    report = ConflictDetector(sampled["conns"]).check(sampled["profile"], sampled["session"])
    assert not report.has_conflict, "a row we never staged is not a conflict"


def test_composite_keys_match_the_right_rows(tmp_path, monkeypatch):
    """stream_by_keys builds (k1 = ? AND k2 = ?) OR ... — it must not cross-match."""
    monkeypatch.setattr(core, "DB_STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(core, "DB_AUDIT_DIR", tmp_path / "audit")
    core.DB_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    core.DB_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    src = tmp_path / "composite.sqlite"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE t (a INTEGER, b INTEGER, v TEXT, PRIMARY KEY (a, b))")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)",
                    [(a, b, f"{a}-{b}") for a in range(3) for b in range(3)])
    con.commit()
    con.close()

    profile = ConnectionProfile(name="t", engine="sqlite", database=str(src))
    importer = StreamingImporter(ConnectionManager())
    wanted = [{"a": 0, "b": 0}, {"a": 1, "b": 2}, {"a": 2, "b": 1}]
    got = [r for chunk in importer.stream_by_keys(profile, "t", ["a", "b"], wanted)
           for r in chunk]

    assert sorted(r["v"] for r in got) == ["0-0", "1-2", "2-1"]


# ------------------------------- audit log -------------------------------

def test_concurrent_appends_do_not_lose_entries(bench):
    """Every call site builds its own AuditLog, and an append rewrites the WHOLE
    (encrypted) file. A per-instance lock guarded nothing across instances, so two
    writers silently dropped each other's entries from the one record of what
    touched the user's real data."""
    from app.database.models import AuditEntry

    def writer(tag):
        log = AuditLog("sess")          # a separate instance per thread, as in the app
        for i in range(20):
            log.append(AuditEntry(session_id="sess", table="people",
                                  column=f"{tag}-{i}", status="ok"))

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert AuditLog("sess").count() == 60


# ------------------------------ connection vault ------------------------------

def test_a_vault_written_at_a_different_kdf_cost_still_opens(tmp_path, monkeypatch):
    """The blob records n/r/p precisely so the cost can change. unlock() ignored them
    and always derived at the module defaults, so raising _SCRYPT_N would have made
    every existing vault permanently undecryptable with no migration path."""
    from app.database import vault as vault_mod
    from app.database.vault import Vault, VaultError

    path = tmp_path / "db_vault.enc"
    monkeypatch.setattr(vault_mod, "_SCRYPT_N", 2 ** 12)   # cheap "old" vault
    v = Vault(path)
    v.unlock("pw")
    v.upsert_profile({"id": "p1", "name": "old", "engine": "sqlite", "password": "s3cret"})

    # The defaults move on; the existing blob must still open with its own cost.
    monkeypatch.setattr(vault_mod, "_SCRYPT_N", 2 ** 13)
    v2 = Vault(path)
    v2.unlock("pw")
    assert v2.get_profile("p1").password == "s3cret"
    with pytest.raises(VaultError):
        Vault(path).unlock("wrong")

    # Changing the password re-encrypts at the CURRENT cost — the upgrade path.
    v2.change_password("pw", "pw2")
    assert json.loads(path.read_text())["n"] == 2 ** 13
    v3 = Vault(path)
    v3.unlock("pw2")
    assert v3.get_profile("p1").password == "s3cret"


def test_a_corrupt_vault_reports_itself_rather_than_500ing(tmp_path):
    from app.database.vault import Vault, VaultError

    path = tmp_path / "db_vault.enc"
    path.write_text("this is not json")
    with pytest.raises(VaultError):
        Vault(path).unlock("pw")
