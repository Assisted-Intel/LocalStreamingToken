#!/usr/bin/env python3
"""
Local Streaming Token — application-wide encryption at rest.

All of the app's data/preference files are encrypted at rest with **AES-256-GCM**.
The scheme mirrors (and shares parameters with) the proven DB connection vault in
``app/database/vault.py``:

  * A random 32-byte **Data Encryption Key (DEK)** encrypts every file. It is
    generated once and never stored in the clear.
  * The DEK is **wrapped** (AES-256-GCM) by a **Key Encryption Key (KEK)** that is
    derived from the user's login password via **scrypt**. Logging in unwraps the
    DEK; a wrong password makes the GCM unwrap fail, so *unlocking is the
    authentication check* — there is no separate password hash.
  * Changing the password re-derives a KEK and re-wraps the *same* DEK, so the bulk
    data never has to be re-encrypted.

Keyfile at rest (``settings/app_key.enc``, JSON with base64 fields)::

    {"version":1,"kdf":"scrypt","n":..,"r":..,"p":..,
     "salt":b64,"wrap_nonce":b64,"wrapped_dek":b64,
     "username":"admin","default_creds":true,"warning_dismissed":false,
     "flask_secret":b64}

Per-file at rest (same filename, so path constants are unchanged)::

    b"LSTENC1\n" + nonce(12) + AESGCM(DEK).encrypt(nonce, plaintext, AAD=b"lst-file-v1")

The unlocked DEK lives in this module's memory only (``_ACTIVE_DEK``) for the life
of the process, set at login and cleared at logout. The ``cryptography`` package is
imported lazily, matching vault.py.
"""

from __future__ import annotations

import base64
import json
import os
import threading

# scrypt work factors — identical to the DB vault so behaviour/cost is consistent.
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32                      # AES-256

MAGIC = b"LSTENC1\n"              # marks an encrypted file (lets migration sniff legacy plaintext)
_AAD_FILE = b"lst-file-v1"        # binds file ciphertext to this app/purpose
_AAD_WRAP = b"lst-keywrap-v1"     # binds the wrapped-DEK ciphertext

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"


class AuthError(Exception):
    """Wrong password / corrupt keyfile."""


# ------------------------------ base64 helpers ------------------------------
def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# ------------------------------ in-memory active key ------------------------------
_KEY_LOCK = threading.RLock()
_ACTIVE_DEK: bytes | None = None


def set_active_key(dek: bytes) -> None:
    global _ACTIVE_DEK
    with _KEY_LOCK:
        _ACTIVE_DEK = dek


def clear_key() -> None:
    global _ACTIVE_DEK
    with _KEY_LOCK:
        _ACTIVE_DEK = None


def get_active_key() -> bytes | None:
    with _KEY_LOCK:
        return _ACTIVE_DEK


def is_unlocked() -> bool:
    with _KEY_LOCK:
        return _ACTIVE_DEK is not None


def dek_hex() -> str:
    """Hex of the active DEK, for use as a DuckDB ``ENCRYPTION_KEY``. Raises if locked."""
    dek = get_active_key()
    if dek is None:
        raise AuthError("No active encryption key (not unlocked).")
    return dek.hex()


# ------------------------------ file encryption ------------------------------
def is_encrypted(blob: bytes) -> bool:
    return isinstance(blob, (bytes, bytearray)) and bytes(blob[:len(MAGIC)]) == MAGIC


def encrypt_bytes(data: bytes, dek: bytes | None = None) -> bytes:
    """Return ``MAGIC + nonce + AESGCM(data)`` using the given (or active) DEK."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if dek is None:
        dek = get_active_key()
    if dek is None:
        raise AuthError("Cannot encrypt: no active encryption key.")
    nonce = os.urandom(12)
    ct = AESGCM(dek).encrypt(nonce, data, _AAD_FILE)
    return MAGIC + nonce + ct


def decrypt_bytes(blob: bytes, dek: bytes | None = None) -> bytes:
    """Reverse :func:`encrypt_bytes`. Raises ``AuthError`` on a wrong key / tamper."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not is_encrypted(blob):
        raise AuthError("Not an encrypted blob (missing header).")
    if dek is None:
        dek = get_active_key()
    if dek is None:
        raise AuthError("Cannot decrypt: no active encryption key.")
    body = bytes(blob[len(MAGIC):])
    nonce, ct = body[:12], body[12:]
    try:
        return AESGCM(dek).decrypt(nonce, ct, _AAD_FILE)
    except Exception as e:
        raise AuthError(f"Decryption failed: {e}")


# ------------------------------ key derivation ------------------------------
def _derive_kek(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    kdf = Scrypt(salt=salt, length=_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive((password or "").encode("utf-8"))


# ------------------------------ keyfile ------------------------------
def keyfile_exists(path) -> bool:
    return os.path.exists(path)


def _read_keyfile(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.loads(fh.read())


def _write_keyfile(path, blob: dict) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(blob, indent=2))
    os.replace(tmp, path)


def _wrap_dek(dek: bytes, password: str) -> dict:
    """Derive a fresh KEK from ``password`` and wrap ``dek`` under it. Returns the
    salt/nonce/ciphertext fields for the keyfile."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt = os.urandom(16)
    kek = _derive_kek(password, salt)
    nonce = os.urandom(12)
    wrapped = AESGCM(kek).encrypt(nonce, dek, _AAD_WRAP)
    return {"salt": _b64e(salt), "wrap_nonce": _b64e(nonce), "wrapped_dek": _b64e(wrapped)}


def create_keyfile(path, password: str = DEFAULT_PASSWORD,
                   username: str = DEFAULT_USERNAME) -> bytes:
    """First-run: generate a random DEK, wrap it under ``password``, persist the
    keyfile (with a random Flask secret), and return the DEK."""
    dek = os.urandom(_KEY_LEN)
    blob = {
        "version": 1, "kdf": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P,
        "username": username,
        "default_creds": (username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD),
        "warning_dismissed": False,
        "flask_secret": _b64e(os.urandom(32)),
    }
    blob.update(_wrap_dek(dek, password))
    _write_keyfile(path, blob)
    return dek


def unlock(path, password: str) -> bytes:
    """Unlock the keyfile with ``password`` and return the DEK.

    Raises ``AuthError`` on a wrong password or corrupt keyfile. Does NOT set the
    active key — the caller decides (so a verification check can unlock without
    activating)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        blob = _read_keyfile(path)
        salt = _b64d(blob["salt"])
        kek = _derive_kek(password, salt)
        dek = AESGCM(kek).decrypt(_b64d(blob["wrap_nonce"]), _b64d(blob["wrapped_dek"]), _AAD_WRAP)
    except AuthError:
        raise
    except Exception:
        raise AuthError("Incorrect password.")
    if len(dek) != _KEY_LEN:
        raise AuthError("Corrupt keyfile.")
    return dek


def verify_username(path, username: str) -> bool:
    """True if ``username`` matches the stored one (case-insensitive)."""
    try:
        stored = _read_keyfile(path).get("username", DEFAULT_USERNAME)
    except Exception:
        return False
    return (username or "").strip().lower() == (stored or "").strip().lower()


def get_username(path) -> str:
    try:
        return _read_keyfile(path).get("username", DEFAULT_USERNAME)
    except Exception:
        return DEFAULT_USERNAME


def change_credentials(path, old_password: str, new_username: str = "",
                       new_password: str = "") -> dict:
    """Verify ``old_password`` (by unwrapping the DEK), then re-wrap the *same* DEK
    under ``new_password`` (if given) and/or update the username. Returns status()."""
    dek = unlock(path, old_password)          # raises AuthError if old password wrong
    blob = _read_keyfile(path)
    pw = new_password if new_password else old_password
    uname = (new_username.strip() if new_username and new_username.strip()
             else blob.get("username", DEFAULT_USERNAME))
    blob["username"] = uname
    blob.update(_wrap_dek(dek, pw))
    # Once the user has set anything other than the defaults, stop nagging.
    if not (uname == DEFAULT_USERNAME and pw == DEFAULT_PASSWORD):
        blob["default_creds"] = False
    _write_keyfile(path, blob)
    return status(path)


def dismiss_default_warning(path) -> dict:
    blob = _read_keyfile(path)
    blob["warning_dismissed"] = True
    _write_keyfile(path, blob)
    return status(path)


def status(path) -> dict:
    """Non-secret keyfile status for the browser / login response."""
    try:
        blob = _read_keyfile(path)
    except Exception:
        return {"exists": False}
    return {
        "exists": True,
        "username": blob.get("username", DEFAULT_USERNAME),
        "using_default_creds": bool(blob.get("default_creds", False)),
        "warning_dismissed": bool(blob.get("warning_dismissed", False)),
    }


def get_flask_secret(path) -> bytes:
    """Return the persisted Flask session-signing secret, creating the keyfile with a
    fresh secret on first run if it does not exist yet."""
    if not keyfile_exists(path):
        create_keyfile(path)                  # first run: default admin/admin keyfile
    blob = _read_keyfile(path)
    secret = blob.get("flask_secret")
    if not secret:
        secret = _b64e(os.urandom(32))
        blob["flask_secret"] = secret
        _write_keyfile(path, blob)
    return _b64d(secret)
