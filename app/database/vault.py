#!/usr/bin/env python3
"""
Encrypted connection vault.

All database connection profiles (with their credentials) are stored in a single
AES-GCM encrypted blob at ``settings/db_vault.enc``. The encryption key is derived
from a user master password via **scrypt** and is held only in memory after the
user unlocks the vault for the session — it is never written to disk. Nothing in
the blob is readable at rest without the password.

At-rest format (JSON, all binary fields base64):
    {"version":1,"kdf":"scrypt","n":..,"r":..,"p":..,"salt":b64,
     "nonce":b64,"ct":b64}

Plaintext (once decrypted) is ``{"profiles":[<ConnectionProfile dict>, ...]}``.

The ``cryptography`` package is imported lazily so the rest of the app boots even
if the optional DB dependencies are not yet installed.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import threading
from typing import Optional

from .models import ConnectionProfile

# scrypt work factors. N=2**15 keeps unlock well under a second on typical
# hardware while staying expensive to brute-force. r/p per common guidance.
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32              # AES-256
_AAD = b"lst-db-vault-v1"  # additional-authenticated-data binds the ciphertext to this app


class VaultError(Exception):
    """Wrong password, corrupt blob, or an operation attempted while locked."""


class VaultLocked(VaultError):
    pass


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


class Vault:
    """Thread-safe, session-scoped encrypted store of connection profiles.

    Lifecycle: ``unlock(password)`` (creates the vault on first use), then
    ``list_profiles`` / ``get_profile`` / ``upsert_profile`` / ``delete_profile``;
    ``lock()`` wipes the key from memory. A single instance is created by the
    Flask app and shared across requests (guarded by an RLock)."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.RLock()
        self._key: Optional[bytes] = None
        self._salt: bytes = b""
        # KDF cost of the CURRENTLY loaded blob. An old vault keeps deriving with the
        # parameters it was written with; only change_password re-encrypts at the
        # module defaults. Without this, raising _SCRYPT_N would make every existing
        # vault permanently undecryptable with no migration path.
        self._kdf: tuple = (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
        self._profiles: dict = {}   # id -> profile dict (with secrets), in-memory only

    # ------------------------------ status ------------------------------
    def exists(self) -> bool:
        return os.path.exists(self._path)

    def is_unlocked(self) -> bool:
        with self._lock:
            return self._key is not None

    def status(self) -> dict:
        with self._lock:
            return {"exists": self.exists(), "unlocked": self.is_unlocked(),
                    "profile_count": len(self._profiles) if self.is_unlocked() else None}

    # ------------------------------ crypto ------------------------------
    @staticmethod
    def _derive(password: str, salt: bytes, kdf_params: tuple = None) -> bytes:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        n, r, p = kdf_params or (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
        kdf = Scrypt(salt=salt, length=_KEY_LEN, n=n, r=r, p=p)
        return kdf.derive((password or "").encode("utf-8"))

    def _encrypt(self, key: bytes, plaintext: bytes) -> dict:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        salt = self._salt  # reuse the salt bound to this key
        n, r, p = self._kdf
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, _AAD)
        # Record the cost this blob was actually written with, so unlock can
        # reproduce the key even after the module defaults move on.
        return {"version": 1, "kdf": "scrypt", "n": n, "r": r,
                "p": p, "salt": _b64e(salt), "nonce": _b64e(nonce), "ct": _b64e(ct)}

    # ------------------------------ unlock/lock ------------------------------
    def unlock(self, password: str) -> dict:
        """Unlock (or, on first use, create) the vault with ``password``.

        Raises ``VaultError`` if the password is wrong or the blob is corrupt.
        Returns ``status()``."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        with self._lock:
            if not self.exists():
                # First run: create an empty encrypted vault with a fresh salt.
                self._salt = os.urandom(16)
                self._kdf = (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
                self._key = self._derive(password, self._salt, self._kdf)
                self._profiles = {}
                self._persist_locked()
                return self.status()

            # A truncated or hand-edited blob raises JSONDecodeError/KeyError/binascii
            # errors, none of which are VaultError — the route would 500 instead of
            # telling the user their vault file is damaged.
            try:
                blob = json.loads(open(self._path, "r", encoding="utf-8").read())
                salt = _b64d(blob["salt"])
                nonce, ct = _b64d(blob["nonce"]), _b64d(blob["ct"])
                kdf = (int(blob.get("n", _SCRYPT_N)), int(blob.get("r", _SCRYPT_R)),
                       int(blob.get("p", _SCRYPT_P)))
            except VaultError:
                raise
            except Exception as e:
                raise VaultError(f"The vault file is unreadable or corrupt: {e}")
            key = self._derive(password, salt, kdf)
            try:
                pt = AESGCM(key).decrypt(nonce, ct, _AAD)
            except Exception:
                raise VaultError("Incorrect master password.")
            try:
                data = json.loads(pt.decode("utf-8"))
            except Exception as e:
                raise VaultError(f"The vault decrypted but its contents are corrupt: {e}")
            self._salt, self._kdf, self._key = salt, kdf, key
            self._profiles = {p["id"]: p for p in data.get("profiles", []) if p.get("id")}
            return self.status()

    def lock(self) -> dict:
        with self._lock:
            self._key = None
            self._profiles = {}
            self._salt = b""
            self._kdf = (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
            return self.status()

    def set_path(self, path) -> dict:
        """Re-point the vault at a different ``db_vault.enc`` (a different data
        profile). Locks first so the previous profile's key/credentials are wiped
        from memory; the next Database use re-unlocks with that profile's password."""
        with self._lock:
            self.lock()
            self._path = path
            return self.status()

    def change_password(self, old_password: str, new_password: str) -> dict:
        with self._lock:
            self._require_unlocked()
            # Verify old password by re-deriving against the stored salt AND the cost
            # this blob was written with (which may predate the current defaults).
            if not hmac.compare_digest(self._derive(old_password, self._salt, self._kdf),
                                       self._key):
                raise VaultError("Current password is incorrect.")
            # New salt + key at the CURRENT cost, then re-encrypt the same profiles.
            # This doubles as the upgrade path for a vault written at an older cost.
            self._salt = os.urandom(16)
            self._kdf = (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
            self._key = self._derive(new_password, self._salt, self._kdf)
            self._persist_locked()
            return self.status()

    # ------------------------------ profiles ------------------------------
    def list_profiles(self) -> list:
        """Masked profiles (no secrets), for the browser."""
        with self._lock:
            self._require_unlocked()
            return [ConnectionProfile.from_dict(p).masked() for p in self._profiles.values()]

    def get_profile(self, profile_id: str) -> ConnectionProfile:
        """Full profile WITH credentials — server-side use only (connections)."""
        with self._lock:
            self._require_unlocked()
            p = self._profiles.get(profile_id)
            if not p:
                raise VaultError(f"No connection profile {profile_id!r}.")
            return ConnectionProfile.from_dict(p)

    def upsert_profile(self, incoming: dict) -> dict:
        """Insert or update a profile. Blank secret fields on an update keep the
        existing stored value (so the masked editor never wipes a saved password)."""
        with self._lock:
            self._require_unlocked()
            prof = ConnectionProfile.from_dict(incoming)
            if not prof.id:
                prof.id = ConnectionProfile().id
            existing = self._profiles.get(prof.id)
            if existing:
                for f in ConnectionProfile.SECRET_FIELDS:
                    if not getattr(prof, f, ""):
                        setattr(prof, f, existing.get(f, ""))
                # Preserve an SSH tunnel secret the same way.
                if prof.ssh_tunnel and existing.get("ssh_tunnel"):
                    for f in ("password", "private_key"):
                        if not prof.ssh_tunnel.get(f):
                            prof.ssh_tunnel[f] = existing["ssh_tunnel"].get(f, "")
                prof.created = existing.get("created", prof.created)
            from .models import _now
            prof.updated = _now()
            self._profiles[prof.id] = prof.to_dict()
            self._persist_locked()
            return prof.masked()

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock:
            self._require_unlocked()
            existed = profile_id in self._profiles
            self._profiles.pop(profile_id, None)
            if existed:
                self._persist_locked()
            return existed

    # ------------------------------ internals ------------------------------
    def _require_unlocked(self):
        if self._key is None:
            raise VaultLocked("The connection vault is locked. Unlock it first.")

    def _persist_locked(self):
        """Encrypt current profiles and atomically write the blob. Caller holds the lock."""
        plaintext = json.dumps({"profiles": list(self._profiles.values())}).encode("utf-8")
        blob = self._encrypt(self._key, plaintext)
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(blob))
        os.replace(tmp, self._path)
