#!/usr/bin/env python3
"""What the launcher decides to bind, before it touches a socket.

``_resolve_bind`` is split out of ``main()`` precisely so the precedence rules — saved
file, overridden by this run's flags — can be checked without starting a server.
"""

from types import SimpleNamespace

import pytest

import main


def args(port=None, lan=None):
    """argparse's result. ``lan`` is a tri-state: None means no flag was given."""
    return SimpleNamespace(port=port, lan=lan)


def cfg(lan_enabled=False, port=8756, explicit=False):
    return {"lan_enabled": lan_enabled, "port": port, "port_is_explicit": explicit}


# --------------------------- host ---------------------------

def test_the_default_is_this_computer_only():
    assert main._resolve_bind(cfg(), args())["host"] == "127.0.0.1"


def test_sharing_binds_every_interface():
    st = main._resolve_bind(cfg(lan_enabled=True), args())
    assert st["host"] == "0.0.0.0" and st["lan"] is True


def test_no_lan_wins_over_the_saved_setting():
    """The escape hatch: start privately once without editing anything."""
    assert main._resolve_bind(cfg(lan_enabled=True), args(lan=False))["host"] == "127.0.0.1"


def test_lan_wins_over_the_saved_setting():
    assert main._resolve_bind(cfg(lan_enabled=False), args(lan=True))["host"] == "0.0.0.0"


def test_no_flag_leaves_the_saved_setting_alone():
    """Why both flags carry default=None: 'not given' has to be distinguishable from
    'explicitly off', or the saved value could never win."""
    assert main._resolve_bind(cfg(lan_enabled=True), args(lan=None))["lan"] is True


# --------------------------- port ---------------------------

def test_the_saved_port_is_used_when_no_flag_is_given():
    assert main._resolve_bind(cfg(port=9000), args())["port"] == 9000


def test_the_flag_overrides_the_saved_port():
    assert main._resolve_bind(cfg(port=9000), args(port=9100))["port"] == 9100


def test_an_untouched_default_port_may_drift_to_a_free_one():
    """Nobody chose 8756, so falling back to a free port when it is taken keeps the
    long-standing behaviour of always starting."""
    assert main._resolve_bind(cfg(), args())["port_is_explicit"] is False


@pytest.mark.parametrize("saved,flag", [(True, None), (False, 9100)])
def test_a_chosen_port_is_marked_so_it_never_drifts(saved, flag):
    """Someone told a colleague an address. Silently moving to another port would
    break it without ever saying so."""
    st = main._resolve_bind(cfg(port=9000, explicit=saved), args(port=flag))
    assert st["port_is_explicit"] is True


# --------------------------- failure messages ---------------------------

class _WinError(Exception):
    """Stands in for the OSError Windows raises; winerror is read-only on the real one."""

    def __init__(self, winerror):
        self.winerror = winerror


def test_a_port_windows_has_reserved_is_explained_not_just_reported():
    """Hyper-V, WSL and Docker reserve whole TCP ranges, so a port nothing is listening
    on still refuses to bind. Without naming that, the user sees a bare PermissionError
    on an apparently idle port and has nowhere to go."""
    msg = main._bind_message("0.0.0.0", 9000, _WinError(10013))
    assert "reserved" in msg and "excludedportrange" in msg


def test_an_occupied_port_says_so_plainly():
    assert "in use" in main._bind_message("0.0.0.0", 9000, _WinError(10048))


def test_an_unrecognised_failure_still_reports_the_address_and_the_error():
    msg = main._bind_message("0.0.0.0", 9000, OSError("something else"))
    assert "0.0.0.0:9000" in msg and "something else" in msg
