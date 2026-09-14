#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Launcher: starts the local web server and opens the default browser to the app.
The GUI now lives entirely in the browser (previously wxPython). Run:

    python main.py

By default the server binds to localhost only. Settings -> Network Access can open it
to other machines on the local network and change the port; those two values live in
plaintext ``settings/network.json`` (app/netconfig.py) because the socket is bound long
before anyone has logged in to decrypt anything else.

All filesystem features (batch a folder, load/save library files) use native OS dialogs
and read/write paths directly on THIS machine — including when the browser driving them
is on another computer. Nothing is uploaded.
"""

import argparse
import getpass
import json
import multiprocessing
import os
import sys
import threading
import urllib.request
import webbrowser

from app.core import APP_NAME, APP_AUTHOR


def _reset_password():
    """`--reset-password`: change the login username/password knowing the CURRENT one.
    Re-wraps the same Data Encryption Key, so all encrypted data is preserved."""
    from app import core, crypto
    kf = core.APP_KEYFILE
    if not crypto.keyfile_exists(kf):
        print("No login has been set up yet — it will be created (admin/admin) on first launch.")
        return
    st = crypto.status(kf)
    print(f"\n  {APP_NAME} — reset login password")
    print(f"  Current username: {st.get('username')}\n")
    current = getpass.getpass("  Current password: ")
    try:
        crypto.unlock(kf, current)
    except crypto.AuthError:
        print("\n  Incorrect password. Your encrypted data cannot be opened without it.")
        print("  If you've forgotten it, run:  python main.py --forgot-password")
        print("  (that erases the encrypted data and resets the login to admin/admin).")
        sys.exit(1)
    new_user = input(f"  New username [{st.get('username')}]: ").strip() or st.get("username")
    while True:
        p1 = getpass.getpass("  New password: ")
        if not p1:
            print("  Password cannot be empty.")
            continue
        if p1 != getpass.getpass("  Confirm new password: "):
            print("  Passwords don't match — try again.")
            continue
        break
    crypto.change_credentials(kf, old_password=current, new_username=new_user, new_password=p1)
    print("\n  Done. Login updated; your data is unchanged.\n")


def _forgot_password():
    """`--forgot-password`: last resort when the password is lost. The encryption key
    is unrecoverable, so this ERASES the encrypted data and recreates a fresh
    admin/admin login. Files you previously exported are unaffected."""
    from app import core, crypto, migrate
    kf = core.APP_KEYFILE
    print(f"\n  {APP_NAME} — forgotten-password reset\n")
    print("  WARNING: the login password wraps the encryption key. Without it, your")
    print("  encrypted data (chats, presets, libraries, personas, …) CANNOT be")
    print("  recovered. This will DELETE that data and reset the login to admin/admin.")
    print("  Anything you already exported is NOT affected.")
    print("")
    print("  NOTE: if RAG is set to the LanceDB vector store, that store is NOT")
    print("  encrypted — its chunk text and embeddings are readable without the")
    print("  password, and are not protected by this guarantee. The DuckDB vector")
    print("  store IS encrypted. See Settings -> RAG -> Vector store.\n")
    if input('  Type "ERASE" to confirm: ').strip() != "ERASE":
        print("\n  Cancelled — nothing was changed.\n")
        return
    removed = migrate.wipe_encrypted()
    if crypto.keyfile_exists(kf):
        os.remove(kf)
    crypto.create_keyfile(kf)   # fresh key, default admin/admin
    print(f"\n  Removed {removed} encrypted file(s). Login reset to admin / admin.")
    print("  Start the app and sign in — you'll be prompted to set a new password.\n")
    print("  NOTE: this does NOT reset settings/network.json. If you had opened the app")
    print("  to your network, it is still open — with the default password back.\n")


# --------------------------- binding ---------------------------

def _resolve_bind(cfg, args):
    """Decide what to bind from the saved config plus this run's flags.

    Pure — no sockets, no printing — so the precedence rules are testable without
    starting a server. ``args.lan`` is a tri-state: None means "no flag given, use the
    saved value", which is why both flags carry ``default=None``.
    """
    lan = cfg.get("lan_enabled", False) if args.lan is None else args.lan
    port = cfg.get("port", 8756) if args.port is None else args.port
    return {
        "host": "0.0.0.0" if lan else "127.0.0.1",
        "port": port,
        "lan": bool(lan),
        # A port the user actually chose must never silently drift; the untouched
        # default still falls back to a free one, as it always has.
        "port_is_explicit": bool(cfg.get("port_is_explicit")) or args.port is not None,
    }


def _is_our_app(port, timeout=0.8):
    """True when the thing already on ``port`` is another copy of this app.

    urllib rather than requests: nothing else on the boot path needs requests, and the
    test suite blocks that module wholesale.
    """
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/auth/status", timeout=timeout) as r:
            return "exists" in json.loads(r.read().decode("utf-8"))
    except Exception:
        return False


def _bind_message(host, port, err):
    """Explain a failed bind in terms of what the user can do about it."""
    win = getattr(err, "winerror", None)
    if win == 10013:
        # Not "in use" — Hyper-V, WSL and Docker reserve whole dynamic TCP blocks, so a
        # port nothing is listening on can still refuse to bind. Without naming this,
        # the user sees an opaque PermissionError on an apparently idle port.
        return (f"  ! Windows has reserved port {port}, so it cannot be used.\n"
                "    List the reserved ranges with:\n"
                "        netsh interface ipv4 show excludedportrange protocol=tcp\n"
                "    then pick a port outside them in Settings -> Network Access.")
    if win == 10048 or getattr(err, "errno", None) in (48, 98):
        return f"  ! Port {port} is already in use by another program."
    return f"  ! Could not bind {host}:{port} — {err}"


def _make_server(host, port, app):
    """``(server, None)`` or ``(None, message)``. Werkzeug's make_server rather than
    app.run() so the Settings page can stop and rebind the socket without killing the
    process — which would drop the in-memory encryption key and log the user out."""
    from werkzeug.serving import make_server
    try:
        return make_server(host, port, app, threaded=True), None
    except OSError as e:
        return None, _bind_message(host, port, e)


def _print_banner(state, host, port, fell_back, auth):
    from app import netconfig
    print(f"\n  {APP_NAME} — by {APP_AUTHOR}")
    print(f"  Serving at http://127.0.0.1:{port}")
    if fell_back and port != state["port"]:
        print(f"  ! Port {state['port']} was unavailable, so this run is on {port} instead.")
    if state["lan"]:
        addresses = netconfig.lan_addresses(refresh=True)
        if addresses:
            print("\n  Also reachable from other computers on your network at:")
            for a in addresses:
                tag = ""
                if a["default_route"]:
                    tag = "   <- try this one first"
                elif a["kind"] == "cgnat":
                    tag = "   (VPN address — only reachable over that VPN)"
                print(f"      http://{a['ip']}:{port}{tag}")
        else:
            print("\n  Network sharing is ON, but no network address was found — this")
            print("  computer may not be connected to a network right now.")
        print("\n  If another computer can't connect, allow Python through Windows")
        print("  Defender Firewall on Private networks (Settings -> Network Access has")
        print("  the exact command).")
        if auth.get("using_default_creds"):
            print("\n  ! WARNING: the login is still admin / admin, and the app is open to")
            print("    your network. Anyone who can reach it can sign in, read your chats")
            print("    and read and write files on this computer. Change the password in")
            print("    Settings, or run:  python main.py --reset-password")


def _serve(app, args):
    """Bind, serve, and rebind whenever Settings asks for a restart.

    One process for the whole session: re-exec'ing would throw away the unwrapped Data
    Encryption Key and force the user to log in again every time they changed a port.
    """
    from app import core, crypto, netconfig

    running = {}

    def stop():
        srv = running.get("srv")
        if srv is not None:
            srv.shutdown()      # returns serve_forever() below; safe from another thread

    netconfig.set_restart_hook(stop)

    live = None                 # what is currently bound, to revert to on a failed rebind
    opened = False
    while True:
        cfg = netconfig.load()
        auth = crypto.status(core.APP_KEYFILE)
        state = _resolve_bind(cfg, args)
        host, port, fell_back = state["host"], state["port"], False

        if port == 0:
            port, fell_back = netconfig.find_free_port(host), True
        elif netconfig.port_in_use(host, port):
            if live is None and _is_our_app(port):
                print(f"\n  {APP_NAME} is already running at http://127.0.0.1:{port}")
                print("  Opening it in your browser.\n")
                webbrowser.open(f"http://127.0.0.1:{port}")
                return
            if state["port_is_explicit"] and live is None:
                print(f"\n  ! Port {port} is already in use by another program.")
                print("    Change it in Settings -> Network Access next time you start,")
                print(f"    or run:  python main.py --port 0\n")
                sys.exit(1)
            # An unchosen default, or a rebind: take a free port rather than refuse to
            # come back up.
            port, fell_back = netconfig.find_free_port(host), True

        srv, err = _make_server(host, port, app)
        if srv is None:
            print("")
            print(err)
            if live is None:
                sys.exit(1)
            print(f"    Staying on http://127.0.0.1:{live[1]} instead.")
            host, port, fell_back = live[0], live[1], True
            srv, err2 = _make_server(host, port, app)
            if srv is None:
                print(err2)
                print("  Could not get the server listening again — stopping.")
                return

        running["srv"] = srv
        live = (host, port)
        netconfig.set_runtime(host, port, state["lan"],
                              requested_port=state["port"], fell_back=fell_back,
                              port_forced=args.port is not None,
                              lan_forced=args.lan is not None)
        _print_banner(state, host, port, fell_back, auth)
        if not opened:
            # Always loopback — http://0.0.0.0 is a bind wildcard, not an address you can
            # open. When the LAN bind is new, Windows' firewall prompt appears at bind
            # time, so give it a moment to land in front rather than behind the browser.
            print("\n  Opening your default browser… (press Ctrl+C to stop)\n")
            threading.Timer(1.8 if state["lan"] else 1.0,
                            lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
            opened = True

        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopping.\n")
            try:
                from app import avatar as avatar_mod
                if avatar_mod.stop_owned():
                    print("  Stopped the Avatar helper we started.\n")
            except Exception:
                pass
            srv.server_close()
            return
        srv.server_close()
        if not netconfig.consume_restart():
            return
        print("\n  Restarting the server with the new network settings…")


def main():
    parser = argparse.ArgumentParser(
        prog="main.py", description=f"{APP_NAME} — local LLM chat server.")
    parser.add_argument("--reset-password", action="store_true",
                        help="Change the login username/password (requires the current password) and exit. Data is preserved.")
    parser.add_argument("--forgot-password", action="store_true",
                        help="Erase the encrypted data and reset the login to admin/admin (use only if the password is lost; unrecoverable).")
    parser.add_argument("--port", type=int, default=None, metavar="N",
                        help="Serve on this port for this run only (0 = any free port). Does not change the saved setting.")
    lan = parser.add_mutually_exclusive_group()
    lan.add_argument("--lan", dest="lan", action="store_true", default=None,
                     help="Open the app to other computers on your network for this run only.")
    lan.add_argument("--no-lan", dest="lan", action="store_false",
                     help="Keep the app on this computer only for this run, whatever the saved setting says.")
    args = parser.parse_args()

    if args.reset_password:
        _reset_password()
        return
    if args.forgot_password:
        _forgot_password()
        return

    from app.server import create_app
    app = create_app()

    # An installed-but-unimportable pandas/numpy silently cripples DuckDB (it retries
    # the failing import for every value it converts). We work around it, but the
    # environment is still broken for anything else that wants those packages.
    from app.core import BROKEN_OPTIONAL_IMPORTS
    if BROKEN_OPTIONAL_IMPORTS:
        print("\n  ! Broken Python packages detected and disabled for this run:")
        for name, err in BROKEN_OPTIONAL_IMPORTS:
            print(f"      {name}: {err}")
        print("    These are installed but cannot be imported, which makes DuckDB")
        print("    dramatically slower. Fix with, e.g.:  pip install -U --force-reinstall numpy pandas")

    _serve(app, args)


if __name__ == "__main__":
    # ingest.extract_many parses documents in a process pool. Under Windows' spawn
    # start method each worker re-runs this file, so a frozen (PyInstaller) build
    # would otherwise relaunch the whole app once per worker.
    multiprocessing.freeze_support()
    main()
