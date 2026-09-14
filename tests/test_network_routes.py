#!/usr/bin/env python3
"""The /api/network routes, the cross-origin guard, and the login brake.

These three exist together for one reason: Settings -> Network Access can put this
server in front of everyone on the local network, and the app was written on the
assumption that only its own operator could reach it.
"""

import json
import socket

import pytest

from app import core, netconfig
from tests.conftest import isolate_paths, make_client


@pytest.fixture(autouse=True)
def reset_netconfig(monkeypatch):
    """No runtime, hook or address cache leaking in from another test."""
    monkeypatch.setattr(netconfig, "_runtime", None)
    monkeypatch.setattr(netconfig, "_restart_hook", None)
    monkeypatch.setattr(netconfig, "_addr_cache", None)
    netconfig._restart_pending.clear()


@pytest.fixture
def free_port():
    """A port nothing is listening on, so the route's occupancy check lets it through."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------- reading ---------------------------

def test_a_fresh_install_reports_itself_as_local_only(client):
    r = client.get("/api/network").get_json()
    assert r["saved"] == {"lan_enabled": False, "port": 8756}
    assert r["restart_required"] is False


def test_there_is_no_live_binding_to_report_when_main_py_did_not_start_us(client):
    """create_app() in a test binds nothing. The card must say so rather than invent a
    binding for the user to compare their unsaved edits against."""
    assert client.get("/api/network").get_json()["runtime"] is None
    assert client.get("/api/network").get_json()["restart_supported"] is False


def test_the_card_is_told_the_password_is_still_the_default(client):
    """The conftest keyfile is admin/admin, which is the state this warning exists for."""
    assert client.get("/api/network").get_json()["using_default_creds"] is True


def test_the_firewall_command_names_the_port_being_saved(client, free_port):
    client.post("/api/network", json={"port": free_port})
    assert str(free_port) in client.get("/api/network").get_json()["firewall_command"]


# --------------------------- writing ---------------------------

def test_saving_is_visible_on_the_next_read(client, free_port):
    r = client.post("/api/network", json={"lan_enabled": True, "port": free_port})
    assert r.status_code == 200
    assert client.get("/api/network").get_json()["saved"] == {
        "lan_enabled": True, "port": free_port}


def test_the_saved_file_is_plaintext_so_the_launcher_can_read_it(client, free_port):
    """Settings written through /api/settings are encrypted with the login password.
    These two cannot be: the socket is bound before anyone logs in."""
    client.post("/api/network", json={"lan_enabled": True, "port": free_port})
    assert json.loads(core.NETWORK_FILE.read_text(encoding="utf-8")) == {
        "lan_enabled": True, "port": free_port}


def test_sharing_is_allowed_while_the_password_is_still_the_default(client):
    """A deliberate product decision: warn loudly, do not block. The card and the
    startup banner both say so, and the user gets to decide."""
    r = client.post("/api/network", json={"lan_enabled": True})
    assert r.status_code == 200
    assert r.get_json()["saved"]["lan_enabled"] is True
    assert r.get_json()["using_default_creds"] is True


@pytest.mark.parametrize("bad", [80, 0, 70000, "abc"])
def test_an_unusable_port_is_refused_and_nothing_is_written(client, bad):
    before = client.get("/api/network").get_json()["saved"]
    r = client.post("/api/network", json={"port": bad})
    assert r.status_code == 400
    assert r.get_json()["error"]
    assert client.get("/api/network").get_json()["saved"] == before


def test_a_port_something_else_is_using_is_refused_before_it_is_saved(client):
    """Saving a port that cannot be bound would only surface as a failure to start at
    the next launch, by which time the Settings page is gone."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(8)
        port = busy.getsockname()[1]
        r = client.post("/api/network", json={"port": port})
        assert r.status_code == 409
        assert "in use" in r.get_json()["error"]
    assert client.get("/api/network").get_json()["saved"]["port"] == 8756


def test_re_saving_the_port_we_are_already_serving_on_is_not_a_conflict(client):
    """The occupant of the live port is us, so probing it must not veto a change to the
    other field."""
    netconfig.set_runtime("127.0.0.1", 8756, False)
    r = client.post("/api/network", json={"lan_enabled": False, "port": 8756})
    assert r.status_code == 200


def test_an_unknown_key_is_ignored_rather_than_stored(client, free_port):
    client.post("/api/network", json={"port": free_port, "not_a_setting": "x"})
    assert "not_a_setting" not in json.loads(core.NETWORK_FILE.read_text(encoding="utf-8"))


def test_a_change_is_reported_as_pending_until_the_server_rebinds(client, free_port):
    netconfig.set_runtime("127.0.0.1", 8756, False)
    assert client.get("/api/network").get_json()["restart_required"] is False
    r = client.post("/api/network", json={"port": free_port}).get_json()
    assert r["restart_required"] is True
    assert r["restarting"] is False        # nothing registered a hook, so nothing rebound


def test_asking_for_a_restart_when_nothing_can_perform_one_says_so(client, free_port):
    r = client.post("/api/network", json={"port": free_port, "restart": True}).get_json()
    assert r["restarting"] is False


# --------------------------- the two config surfaces stay separate ---------

def test_network_keys_posted_to_the_settings_route_are_not_stored_there(client):
    """A future contributor adding these to DEFAULT_SETTINGS would produce a second,
    encrypted copy that main.py can never read and that silently wins in the UI."""
    cfg = client.post("/api/settings", json={"lan_enabled": True, "port": 9999}
                      ).get_json()["config"]
    assert "lan_enabled" not in cfg and "port" not in cfg


def test_the_routes_are_behind_the_login_gate(tmp_path, monkeypatch):
    isolate_paths(tmp_path, monkeypatch)
    from app import server
    c = server.create_app().test_client()
    assert c.get("/api/network").status_code == 401
    assert c.post("/api/network", json={"port": 9000}).status_code == 401


# --------------------------- cross-origin guard ---------------------------

def test_a_post_from_another_site_is_refused(client):
    """Every route parses its body with get_json(force=True), so Content-Type is no
    barrier to a forged cross-site POST once the app is reachable over the network."""
    r = client.post("/api/network", json={"port": 9000},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_the_guard_covers_the_login_itself(client):
    r = client.post("/api/login", json={"username": "admin", "password": "admin"},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_the_apps_own_page_is_not_refused(client, free_port):
    """Same-origin is judged by comparing Origin to the Host the browser used, so this
    holds whatever address the user reached the server by."""
    r = client.post("/api/network", json={"port": free_port},
                    headers={"Origin": "http://localhost"})
    assert r.status_code == 200


def test_reading_is_never_blocked_by_the_guard(client):
    assert client.get("/api/network", headers={"Origin": "http://evil.example"}
                      ).status_code == 200


# --------------------------- login brake ---------------------------

def test_guessing_the_password_is_throttled(tmp_path, monkeypatch):
    """One shared password, and the login page can be put in front of a whole network.
    scrypt costs about 100 ms an attempt, which on its own is not much of a brake."""
    c = make_client(tmp_path, monkeypatch)
    for _ in range(10):
        assert c.post("/api/login", json={"username": "admin", "password": "no"}
                      ).status_code == 401
    r = c.post("/api/login", json={"username": "admin", "password": "no"})
    assert r.status_code == 429
    # Even the right password is held off — the brake is on the address, not the guess.
    assert c.post("/api/login", json={"username": "admin", "password": "admin"}
                  ).status_code == 429


def test_a_successful_sign_in_clears_the_count(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    for _ in range(9):
        c.post("/api/login", json={"username": "admin", "password": "no"})
    assert c.post("/api/login", json={"username": "admin", "password": "admin"}
                  ).status_code == 200
    for _ in range(9):
        assert c.post("/api/login", json={"username": "admin", "password": "no"}
                      ).status_code == 401
