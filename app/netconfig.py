#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Where the HTTP server binds: loopback only (the default) or every interface, and on
which port. Owns ``settings/network.json``.

Why this is not a normal setting
--------------------------------
Every other preference lives in ``settings/profiles/<id>/settings.json``, which is
AES-GCM encrypted and unreadable until the user logs in. A listening socket has to be
bound *before* anyone can log in, so the bind address cannot live there. This file is
therefore PLAINTEXT and APP-WIDE, exactly like the profile registries
(``core.load_json_plain``) and for the same reason. It holds no secrets — a host and a
port — and it is covered by the ``settings/*`` gitignore rule.

Nothing here may raise on the boot path: a hand-edited or corrupt file degrades to the
loopback defaults rather than stopping the app from starting.

IPv4 only. Werkzeug binding ``0.0.0.0`` covers every IPv4 interface; IPv6 needs a
separate ``::`` bind and is deliberately out of scope, not an oversight.
"""

import ipaddress
import socket
import threading
import time

from . import core

# The saved shape. Three keys, nothing else.
DEFAULTS = {
    "lan_enabled": False,
    "port": 8756,
}

# Below 1024 collides with http.sys (80/443) and SMB (445) on Windows and needs root
# elsewhere. Above 49151 is the ephemeral range, where an outgoing connection can claim
# the port before we start — allowed, but warned about.
MIN_PORT, MAX_PORT = 1024, 65535
EPHEMERAL_FROM = 49152

# Set once by main.py immediately before the server binds, so the UI can tell what is
# SAVED from what is actually LIVE. Stays None when the app was not started through
# main.py (every test, and any embedding of create_app()).
_runtime = None

# Restart handshake. main.py registers a hook that stops the running server; the route
# calls request_restart(), main.py's supervise loop then rebinds with the new config.
_restart_hook = None
_restart_pending = threading.Event()

# lan_addresses() resolves a hostname, which can block for seconds on a machine with an
# unreachable DNS server, so the answer is cached until explicitly refreshed.
_addr_cache = None
_addr_cache_at = 0.0
_ADDR_TTL = 60.0


# --------------------------- load / save ---------------------------

def clean_port(value, *, allow_zero=False):
    """Coerce ``value`` to a usable port. Raises ValueError with a message meant to be
    shown to the user — this is one setting where silently dropping a bad value (the
    /api/settings house style) would strand someone on a port they didn't choose."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError("Port must be a whole number.")
    if port == 0 and allow_zero:
        return 0
    if port < MIN_PORT:
        raise ValueError("Ports below %d are reserved for system services." % MIN_PORT)
    if port > MAX_PORT:
        raise ValueError("The highest possible port is %d." % MAX_PORT)
    return port


def limits():
    """The port rules, for the UI. Sent to the browser rather than hard-coded there so
    the input bounds and the ephemeral-range warning cannot drift from what validate()
    actually enforces."""
    return {"min": MIN_PORT, "max": MAX_PORT, "ephemeral_from": EPHEMERAL_FROM}


def validate(patch):
    """Return ``(clean, errors)`` for a partial update. Unknown keys are dropped, the
    way the /api/settings allowlist does."""
    clean, errors = {}, []
    if "lan_enabled" in patch:
        clean["lan_enabled"] = bool(patch["lan_enabled"])
    if "port" in patch:
        try:
            clean["port"] = clean_port(patch["port"])
        except ValueError as e:
            errors.append(str(e))
    return clean, errors


def load():
    """The saved config, with every value validated. Never raises: this runs on the
    boot path, and the file is plaintext and therefore hand-editable."""
    raw = core.load_json_plain(core.NETWORK_FILE, {})
    if not isinstance(raw, dict):
        raw = {}
    cfg = dict(DEFAULTS)
    cfg["lan_enabled"] = bool(raw.get("lan_enabled", DEFAULTS["lan_enabled"]))
    try:
        cfg["port"] = clean_port(raw.get("port", DEFAULTS["port"]))
    except ValueError:
        cfg["port"] = DEFAULTS["port"]
    # Whether the user ever chose a port decides what happens when it is busy: a chosen
    # port fails loudly, the untouched default drifts to a free one (the old behaviour).
    cfg["port_is_explicit"] = "port" in raw
    return cfg


def save(patch):
    """Validate and merge ``patch`` into the saved config. Raises ValueError listing
    every problem, so nothing is half-written."""
    clean, errors = validate(patch)
    if errors:
        raise ValueError(" ".join(errors))
    cfg = load()
    cfg.pop("port_is_explicit", None)
    cfg.update(clean)
    core.save_json_plain(core.NETWORK_FILE, cfg)
    return load()


def bind_host(cfg=None):
    """The address to bind. ``0.0.0.0`` is a wildcard, never something to display —
    print loopback plus lan_addresses() instead."""
    cfg = load() if cfg is None else cfg
    return "0.0.0.0" if cfg.get("lan_enabled") else "127.0.0.1"


# --------------------------- runtime mirror ---------------------------

def set_runtime(host, port, lan_enabled, requested_port=None, fell_back=False,
                port_forced=False, lan_forced=False):
    """``*_forced`` marks a value pinned by a --port/--lan flag for this run. Saving a
    different one in Settings is not something a restart can converge on, so the UI has
    to be told rather than left promising a restart that changes nothing."""
    global _runtime
    _runtime = {"host": host, "port": port, "lan_enabled": bool(lan_enabled),
                "requested_port": requested_port if requested_port is not None else port,
                "fell_back": bool(fell_back),
                "port_forced": bool(port_forced), "lan_forced": bool(lan_forced)}


def runtime():
    """What the server actually bound, or None when it wasn't started via main.py."""
    return dict(_runtime) if _runtime else None


def restart_required(cfg=None):
    """True when a restart would actually change something. A dimension pinned by a
    command-line flag is skipped: no amount of restarting will move it this run."""
    rt = runtime()
    if not rt:
        return False
    cfg = load() if cfg is None else cfg
    if not rt.get("port_forced") and rt["port"] != cfg["port"]:
        return True
    if not rt.get("lan_forced") and rt["host"] != bind_host(cfg):
        return True
    return False


def next_port(cfg=None):
    """The port the server will come back on after a restart — which is the live one,
    not the saved one, when --port pinned it. The page follows this to find the app
    again, so a wrong answer sends the browser to a dead address."""
    rt = runtime()
    cfg = load() if cfg is None else cfg
    if rt and rt.get("port_forced"):
        return rt["port"]
    return cfg["port"]


# --------------------------- restart handshake ---------------------------

def set_restart_hook(fn):
    """main.py registers the callable that stops the running server. Without it the
    server cannot rebind itself and the UI asks for a manual restart instead."""
    global _restart_hook
    _restart_hook = fn


def restart_supported():
    return _restart_hook is not None


def request_restart(delay=0.4):
    """Ask main.py's supervise loop to rebind. Deferred onto a timer so the HTTP
    response that triggered it is flushed to the browser before the socket closes."""
    if _restart_hook is None:
        return False
    if _restart_pending.is_set():
        # One is already scheduled. A second timer would fire after the loop had already
        # rebound, stopping the NEW server with no pending flag left to restart it —
        # i.e. a double-click on Save & restart would quit the app.
        return True
    _restart_pending.set()
    threading.Timer(delay, _restart_hook).start()
    return True


def consume_restart():
    """True exactly once per request_restart(); the supervise loop's exit condition."""
    if _restart_pending.is_set():
        _restart_pending.clear()
        return True
    return False


# --------------------------- port occupancy ---------------------------

def port_in_use(host, port, timeout=0.35):
    """True when something is already listening on ``port``.

    A CONNECT probe, deliberately, not a bind probe. On Windows ``SO_REUSEADDR`` means
    what POSIX calls ``SO_REUSEPORT``: a bind probe carrying it succeeds against a port
    another process is actively listening on, so it reports "free" for a port that is
    plainly in use. Werkzeug's own server sets ``allow_reuse_address``, so the real bind
    would then also succeed and two servers would split accepts on one port.

    A wildcard bind has no address to connect to, so probe loopback for it — anything
    holding ``0.0.0.0:port`` answers there too.
    """
    probe = "127.0.0.1" if host in ("0.0.0.0", "", None, "::") else host
    try:
        with socket.create_connection((probe, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_free_port(host):
    """An OS-assigned free port, for the 'any port will do' path."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


# --------------------------- LAN address detection ---------------------------

def _default_route_ip():
    """The source address the OS would use to reach the outside world — i.e. the NIC a
    peer on the network will actually reach us on.

    A UDP ``connect()`` sends no packets; it only consults the routing table. The target
    is TEST-NET-1 (RFC 5737, reserved for documentation): it works with no internet
    connection at all, and it never looks like the app is phoning home.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return None


def _hostname_ips(timeout=1.5):
    """Every IPv4 the machine's own hostname resolves to.

    ``getaddrinfo`` takes no timeout and can block for tens of seconds when the
    configured DNS server is unreachable, so it runs on a daemon thread we simply stop
    waiting for. A slow resolver costs us a shorter list, never a hung request.
    """
    result = []

    def resolve():
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None,
                                           family=socket.AF_INET):
                result.append(info[4][0])
        except OSError:
            pass

    t = threading.Thread(target=resolve, daemon=True)
    t.start()
    t.join(timeout)
    return list(result)


def _classify(ip):
    """'lan' | 'cgnat' | 'other', or None for an address no peer can use.

    Virtual adapters (WSL2's 172.x, Docker, Hyper-V, VirtualBox) are indistinguishable
    from a real LAN address by IP alone — the stdlib exposes no adapter names, and
    parsing ipconfig is locale-dependent and still ambiguous. So every candidate is
    listed and the UI explains how to pick, rather than guessing wrong.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if addr.is_loopback or addr.is_unspecified or addr.is_multicast:
        return None
    if addr.is_link_local:          # 169.254.x — a disconnected adapter, never reachable
        return None
    if addr in ipaddress.ip_network("100.64.0.0/10"):
        return "cgnat"              # Tailscale / carrier NAT: only up on that VPN
    if addr.is_private:
        return "lan"
    return "other"


def lan_addresses(refresh=False):
    """Addresses another machine could use to reach this one, best candidate first.

    ``[{"ip": "192.168.1.42", "kind": "lan", "default_route": True}, ...]``
    """
    global _addr_cache, _addr_cache_at
    if not refresh and _addr_cache is not None and (time.time() - _addr_cache_at) < _ADDR_TTL:
        return [dict(a) for a in _addr_cache]

    primary = _default_route_ip()
    found, seen = [], set()
    for ip in ([primary] if primary else []) + _hostname_ips():
        if ip in seen:
            continue
        seen.add(ip)
        kind = _classify(ip)
        if kind:
            found.append({"ip": ip, "kind": kind, "default_route": ip == primary})
    # The default-route address first (it is the one that usually works), the rest in a
    # stable order so the list doesn't reshuffle between renders.
    found.sort(key=lambda a: (not a["default_route"], a["kind"] != "lan", a["ip"]))
    _addr_cache, _addr_cache_at = found, time.time()
    return [dict(a) for a in found]


def firewall_command(port):
    """The PowerShell rule that lets other machines through Windows Defender Firewall.

    Handed to the user to copy rather than run for them: it needs elevation, fails
    silently without it, and an app that quietly elevates itself to edit the firewall is
    not one anybody should trust.
    """
    return ('New-NetFirewallRule -DisplayName "Local Streaming Token" '
            '-Direction Inbound -Action Allow -Protocol TCP -LocalPort %d '
            '-Profile Private' % port)
