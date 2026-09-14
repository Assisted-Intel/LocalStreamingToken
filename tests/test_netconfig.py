#!/usr/bin/env python3
"""Where the server binds, and how it works that out.

``settings/network.json`` is the one config file that is PLAINTEXT and hand-editable —
it has to be, since main.py reads it before the encryption key exists. Everything here
follows from that: nothing may raise on a malformed file, and a value that would leave
the user unable to reach their own app is rejected loudly rather than clamped quietly.
"""

import json

import pytest

from app import core, netconfig


@pytest.fixture(autouse=True)
def isolate_network(tmp_path, monkeypatch):
    """Own file, and none of the module globals carried over from another test."""
    monkeypatch.setattr(core, "NETWORK_FILE", tmp_path / "network.json")
    monkeypatch.setattr(netconfig, "_runtime", None)
    monkeypatch.setattr(netconfig, "_restart_hook", None)
    monkeypatch.setattr(netconfig, "_addr_cache", None)
    netconfig._restart_pending.clear()


# --------------------------- load / save ---------------------------

def test_a_fresh_install_is_loopback_only():
    cfg = netconfig.load()
    assert cfg["lan_enabled"] is False
    assert cfg["port"] == 8756
    assert netconfig.bind_host(cfg) == "127.0.0.1"


def test_saving_round_trips_and_lands_as_readable_plaintext():
    """The whole design rests on main.py being able to read this before login, so the
    file must be plain JSON — not the LSTENC1 blob every other settings file is."""
    netconfig.save({"lan_enabled": True, "port": 9000})
    cfg = netconfig.load()
    assert (cfg["lan_enabled"], cfg["port"]) == (True, 9000)
    assert netconfig.bind_host(cfg) == "0.0.0.0"
    on_disk = json.loads(core.NETWORK_FILE.read_text(encoding="utf-8"))
    assert on_disk == {"lan_enabled": True, "port": 9000}


def test_a_partial_save_leaves_the_other_key_alone():
    netconfig.save({"port": 9000})
    netconfig.save({"lan_enabled": True})
    assert netconfig.load()["port"] == 9000


def test_a_hand_edited_mess_falls_back_instead_of_stopping_the_app():
    """This runs on the boot path. A file someone edited badly must cost them their
    custom port, not their ability to start the app."""
    core.NETWORK_FILE.write_text(
        json.dumps({"port": "abc", "lan_enabled": "yes", "junk": 1}), encoding="utf-8")
    cfg = netconfig.load()
    assert cfg["port"] == 8756
    assert cfg["lan_enabled"] is True          # a non-empty string is truthy, and bool()
    assert "junk" not in cfg                   # unknown keys never reach the app


def test_a_file_that_is_not_even_a_dict_loads_as_defaults():
    core.NETWORK_FILE.write_text("[1, 2, 3]", encoding="utf-8")
    assert netconfig.load()["port"] == 8756


def test_choosing_a_port_is_remembered_as_a_choice():
    """A port the user picked must fail loudly when it is taken; the untouched default
    still drifts to a free one. main.py needs to tell those apart."""
    assert netconfig.load()["port_is_explicit"] is False
    netconfig.save({"port": 9000})
    assert netconfig.load()["port_is_explicit"] is True


# --------------------------- validation ---------------------------

@pytest.mark.parametrize("bad", [0, 80, 1023, 70000, "abc", None, ""])
def test_unusable_ports_are_refused_with_a_reason(bad):
    with pytest.raises(ValueError) as e:
        netconfig.clean_port(bad)
    assert str(e.value)


def test_port_zero_is_only_allowed_where_any_free_port_makes_sense():
    assert netconfig.clean_port(0, allow_zero=True) == 0


def test_a_rejected_port_is_not_half_saved():
    netconfig.save({"port": 9000})
    with pytest.raises(ValueError):
        netconfig.save({"lan_enabled": True, "port": 80})
    cfg = netconfig.load()
    assert (cfg["port"], cfg["lan_enabled"]) == (9000, False)


def test_an_ephemeral_port_is_allowed_rather_than_refused():
    """It works; it is just occasionally stolen by an outgoing connection. The UI warns,
    using the same bounds the route enforces — which is why they are published."""
    assert netconfig.clean_port(50000) == 50000
    lim = netconfig.limits()
    assert lim["min"] == netconfig.MIN_PORT and lim["max"] == netconfig.MAX_PORT
    assert lim["min"] < lim["ephemeral_from"] < lim["max"]


def test_unknown_keys_are_dropped_like_the_settings_allowlist():
    clean, errors = netconfig.validate({"port": 9000, "not_a_setting": "x"})
    assert clean == {"port": 9000} and errors == []


# --------------------------- address detection ---------------------------

def _fake_ips(monkeypatch, primary, others):
    monkeypatch.setattr(netconfig, "_default_route_ip", lambda: primary)
    monkeypatch.setattr(netconfig, "_hostname_ips", lambda timeout=1.5: list(others))
    monkeypatch.setattr(netconfig, "_addr_cache", None)


def test_only_addresses_another_machine_could_use_are_offered(monkeypatch):
    """Loopback is not shareable, and 169.254.x means an adapter that failed to get a
    lease — offering either would send the user off chasing an address that can never
    work."""
    _fake_ips(monkeypatch, "192.168.1.42",
              ["127.0.0.1", "169.254.10.5", "0.0.0.0", "192.168.1.42"])
    ips = [a["ip"] for a in netconfig.lan_addresses()]
    assert ips == ["192.168.1.42"]


def test_the_default_route_address_is_offered_first(monkeypatch):
    _fake_ips(monkeypatch, "192.168.1.42", ["172.28.144.1", "10.0.0.9"])
    addrs = netconfig.lan_addresses()
    assert addrs[0] == {"ip": "192.168.1.42", "kind": "lan", "default_route": True}
    assert {a["ip"] for a in addrs} == {"192.168.1.42", "172.28.144.1", "10.0.0.9"}


def test_a_vpn_address_is_labelled_rather_than_hidden(monkeypatch):
    """Tailscale and carrier NAT live in 100.64/10. It is a real address that really
    works — but only over that VPN, so it is listed with a warning, not dropped."""
    _fake_ips(monkeypatch, "100.94.7.2", ["192.168.1.42"])
    kinds = {a["ip"]: a["kind"] for a in netconfig.lan_addresses()}
    assert kinds == {"100.94.7.2": "cgnat", "192.168.1.42": "lan"}


def test_detection_survives_a_machine_with_no_network(monkeypatch):
    _fake_ips(monkeypatch, None, [])
    assert netconfig.lan_addresses() == []


def test_real_detection_never_raises_and_never_offers_loopback():
    for a in netconfig.lan_addresses(refresh=True):
        assert not a["ip"].startswith(("127.", "169.254."))


# --------------------------- runtime + restart ---------------------------

def test_nothing_needs_restarting_when_no_server_is_running():
    """create_app() outside main.py — every test, and any embedded use. The UI has no
    live binding to compare against, so it must not claim one is stale."""
    assert netconfig.runtime() is None
    assert netconfig.restart_required() is False


def test_a_changed_port_is_reported_as_needing_a_restart():
    netconfig.set_runtime("127.0.0.1", 8756, False)
    assert netconfig.restart_required() is False
    netconfig.save({"port": 9000})
    assert netconfig.restart_required() is True


def test_turning_sharing_on_needs_a_restart_even_on_the_same_port():
    netconfig.set_runtime("127.0.0.1", 8756, False)
    netconfig.save({"lan_enabled": True})
    assert netconfig.restart_required() is True


def test_a_restart_cannot_be_requested_when_nothing_can_perform_one():
    assert netconfig.restart_supported() is False
    assert netconfig.request_restart() is False
    assert netconfig.consume_restart() is False


def test_a_restart_fires_the_hook_once():
    calls = []
    netconfig.set_restart_hook(lambda: calls.append(1))
    assert netconfig.request_restart(delay=0.01) is True
    assert netconfig.consume_restart() is True
    assert netconfig.consume_restart() is False


# --------------------------- port occupancy ---------------------------

def test_a_listening_socket_is_seen_as_in_use():
    """A CONNECT probe, not a bind probe: on Windows SO_REUSEADDR lets a bind probe
    succeed against a port another process is actively listening on, which is how you
    end up with two servers splitting accepts on one port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)   # room for both probes below: each leaves a queued connection
        port = srv.getsockname()[1]
        assert netconfig.port_in_use("127.0.0.1", port) is True
        assert netconfig.port_in_use("0.0.0.0", port) is True
    assert netconfig.port_in_use("127.0.0.1", port) is False


def test_asking_twice_schedules_only_one_restart():
    """A double-click on Save and restart used to be fatal: the second timer fired after
    the loop had already rebound, stopping the NEW server with no pending flag left to
    bring it back, so the app simply exited."""
    calls = []
    netconfig.set_restart_hook(lambda: calls.append(1))
    assert netconfig.request_restart(delay=5) is True
    assert netconfig.request_restart(delay=5) is True
    assert netconfig.consume_restart() is True
    assert netconfig.consume_restart() is False
