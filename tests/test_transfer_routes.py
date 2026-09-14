#!/usr/bin/env python3
"""Getting files in and out of the app from a browser that is not on this machine.

Every file feature was built around a native dialog that opens on the server's own
desktop. That is invisible to a phone, so these routes supply the other half: upload to a
staging area and hand the paths to the route that expected a picker, or write to a
staging file and hand back a download link.

The properties worth pinning are the ones that would fail silently: a remote request must
never open a window on the host, and a client must not be able to name a path of its own
choosing where a picker's output was expected.
"""

import io
import json

import pytest

from app import native_dialog, transfer
from tests.conftest import isolate_paths


@pytest.fixture(autouse=True)
def no_native_dialogs(monkeypatch):
    """A dialog opened during these tests would block for 300 seconds on a desktop
    nobody is watching. Make reaching for one a loud failure instead."""
    def refuse(*a, **k):
        raise AssertionError("a native dialog was opened when it should not have been")

    monkeypatch.setattr(native_dialog, "pick_files", refuse)
    monkeypatch.setattr(native_dialog, "save_file", refuse)
    monkeypatch.setattr(native_dialog, "pick_folder", refuse)


REMOTE = {"REMOTE_ADDR": "192.168.1.50"}       # a phone on the LAN


def upload(client, name, data, **kw):
    return client.post("/api/uploads", data={
        "files": (io.BytesIO(data if isinstance(data, bytes) else data.encode()), name),
    }, content_type="multipart/form-data", **kw)


# --------------------------- uploads ---------------------------

def test_an_uploaded_file_lands_somewhere_the_app_can_read(client):
    paths = upload(client, "notes.txt", "hello from a phone").get_json()["paths"]
    assert len(paths) == 1
    from pathlib import Path
    assert Path(paths[0]).read_text(encoding="utf-8") == "hello from a phone"


def test_the_extension_survives_because_everything_downstream_dispatches_on_it(client):
    paths = upload(client, "report.PDF", b"%PDF-1.4").get_json()["paths"]
    assert paths[0].lower().endswith(".pdf")


def test_uploading_nothing_is_an_error_rather_than_an_empty_success(client):
    r = client.post("/api/uploads", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_two_files_of_the_same_name_do_not_overwrite_each_other(client):
    r = client.post("/api/uploads", data={"files": [
        (io.BytesIO(b"first"), "same.txt"),
        (io.BytesIO(b"second"), "same.txt"),
    ]}, content_type="multipart/form-data")
    paths = r.get_json()["paths"]
    from pathlib import Path
    assert len(set(paths)) == 2
    assert {Path(p).read_text() for p in paths} == {"first", "second"}


def test_a_hostile_filename_cannot_escape_the_staging_area(client):
    paths = upload(client, "../../evil.txt", "nope").get_json()["paths"]
    assert transfer.is_staged(paths[0])


# --------------------------- paths supplied by the client ---------------------------

def test_a_route_given_staged_paths_never_opens_a_dialog(client):
    """The autouse fixture turns any dialog into a failure, so reaching this route's
    normal import path at all is the assertion."""
    paths = upload(client, "chats.json", json.dumps(
        {"chats": [{"id": "c1", "title": "Imported", "messages": []}]})).get_json()["paths"]
    r = client.post("/api/chats/import", json={"paths": paths})
    assert r.status_code == 200
    assert r.get_json().get("cancelled") is not True


def test_a_path_the_client_invented_is_ignored(tmp_path, client):
    """Otherwise anyone signed in could name a file anywhere on the machine and have the
    server read it back to them through an import route."""
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    assert transfer.accept_paths([str(secret)]) == []
    # Sent from a phone, so there is no dialog to fall back to: the route reports nothing
    # picked rather than reading the file it was pointed at.
    r = client.post("/api/chats/import", json={"paths": [str(secret)]}, environ_base=REMOTE)
    assert r.get_json().get("cancelled") is True


def test_a_remote_request_never_opens_a_dialog_on_the_host(client):
    """The safety net that does not depend on the client behaving: even a stale page
    posting the old bodyless request must not put a window on someone else's desktop."""
    r = client.post("/api/chats/import", json={}, environ_base=REMOTE)
    assert r.get_json().get("cancelled") is True


def test_choosing_a_folder_remotely_explains_itself_instead_of_hanging(client):
    """A browser cannot hand over a directory, and the picker would open on the host."""
    r = client.post("/api/pick-folder", json={"title": "x"}, environ_base=REMOTE)
    assert r.status_code == 400
    assert "computer running the app" in r.get_json()["error"]


# --------------------------- downloads ---------------------------

def test_an_export_can_come_back_as_a_download(client):
    r = client.post("/api/chats/export", json={"scope": "all", "download": True})
    link = r.get_json()["download"]
    got = client.get(link)
    assert got.status_code == 200
    assert "attachment" in got.headers["Content-Disposition"]
    assert json.loads(got.get_data(as_text=True)).get("chats") is not None


def test_an_export_from_a_remote_browser_is_a_download_without_being_asked(client):
    """A phone has no Save dialog to fall back to, so the choice is made for it."""
    r = client.post("/api/chats/export", json={"scope": "all"}, environ_base=REMOTE)
    assert r.get_json().get("download")


def test_a_download_link_works_once(client):
    link = client.post("/api/chats/export",
                       json={"scope": "all", "download": True}).get_json()["download"]
    assert client.get(link).status_code == 200
    assert client.get(link).status_code == 404


def test_an_unknown_token_is_a_404_not_a_traceback(client):
    assert client.get("/api/download/deadbeef").status_code == 404


def test_the_download_keeps_the_extension_the_route_appended(client):
    """The export routes add ".json" after they are handed a destination; the token has
    to follow, or the browser is offered a path nothing was written to."""
    link = client.post("/api/chats/export",
                       json={"scope": "all", "download": True}).get_json()["download"]
    assert ".json" in client.get(link).headers["Content-Disposition"]


# --------------------------- the local case is untouched ---------------------------

def test_a_local_export_still_uses_the_save_dialog(client, monkeypatch):
    """The desktop keeps its file picker: nothing here is allowed to turn a local install
    into one that silently drops files into the Downloads folder."""
    called = {}

    def fake_save(**kw):
        called["kw"] = kw
        return str(transfer.STAGING_ROOT / "chosen.json")

    monkeypatch.setattr(native_dialog, "save_file", fake_save)
    transfer.STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    r = client.post("/api/chats/export", json={"scope": "all"})
    assert called, "the native Save dialog was not used for a local request"
    assert "download" not in r.get_json()


def test_the_routes_need_a_login(tmp_path, monkeypatch):
    isolate_paths(tmp_path, monkeypatch)
    from app import server
    c = server.create_app().test_client()
    assert c.post("/api/uploads", data={}).status_code == 401
    assert c.get("/api/download/x").status_code == 401
