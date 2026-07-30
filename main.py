#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Launcher: starts the local web server and opens the default browser to the app.
The GUI now lives entirely in the browser (previously wxPython). Run:

    python main.py

The server binds to localhost only. All filesystem features (batch a folder,
load/save library files) use native OS dialogs and read/write paths directly on
this machine — nothing is uploaded.
"""

import argparse
import getpass
import os
import socket
import sys
import threading
import webbrowser

from app.core import APP_NAME, APP_AUTHOR

HOST = "127.0.0.1"
PREFERRED_PORT = 8756


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
    print("  encrypted data (chats, presets, libraries, personas, RAG store, …) CANNOT")
    print("  be recovered. This will DELETE that data and reset the login to admin/admin.")
    print("  Anything you already exported is NOT affected.\n")
    if input('  Type "ERASE" to confirm: ').strip() != "ERASE":
        print("\n  Cancelled — nothing was changed.\n")
        return
    removed = migrate.wipe_encrypted()
    if crypto.keyfile_exists(kf):
        os.remove(kf)
    crypto.create_keyfile(kf)   # fresh key, default admin/admin
    print(f"\n  Removed {removed} encrypted file(s). Login reset to admin / admin.")
    print("  Start the app and sign in — you'll be prompted to set a new password.\n")


def _find_free_port(host, preferred):
    """Return the preferred port if free, otherwise an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(
        prog="main.py", description=f"{APP_NAME} — local LLM chat server.")
    parser.add_argument("--reset-password", action="store_true",
                        help="Change the login username/password (requires the current password) and exit. Data is preserved.")
    parser.add_argument("--forgot-password", action="store_true",
                        help="Erase the encrypted data and reset the login to admin/admin (use only if the password is lost; unrecoverable).")
    args = parser.parse_args()

    if args.reset_password:
        _reset_password()
        return
    if args.forgot_password:
        _forgot_password()
        return

    from app.server import create_app
    port = _find_free_port(HOST, PREFERRED_PORT)
    url = f"http://{HOST}:{port}"
    app = create_app()

    # Open the browser shortly after the server starts accepting connections.
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"\n  {APP_NAME} — by {APP_AUTHOR}")
    print(f"  Serving at {url}")
    print(f"  Opening your default browser… (press Ctrl+C to stop)\n")

    # threaded=True so streaming responses don't block other requests (and the
    # native-dialog subprocess). use_reloader=False so we don't open two browsers.
    app.run(host=HOST, port=port, threaded=True, use_reloader=False, debug=False)


if __name__ == "__main__":
    main()
