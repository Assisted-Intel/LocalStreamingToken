#!/usr/bin/env python3
"""Tests for the at-rest encryption layer (app/crypto.py) and the encrypted file
I/O funnel in app/core.py. These exercise only the crypto primitives + JSON helpers,
so they need `cryptography` but neither Ollama nor DuckDB."""

import importlib
from pathlib import Path

import pytest

from app import crypto, core


@pytest.fixture(autouse=True)
def _clear_key():
    """Every test starts and ends locked, so state never leaks between tests."""
    crypto.clear_key()
    yield
    crypto.clear_key()


# ------------------------------ primitives ------------------------------
def test_encrypt_decrypt_roundtrip():
    dek = b"\x11" * 32
    blob = crypto.encrypt_bytes(b"hello world", dek)
    assert crypto.is_encrypted(blob)
    assert blob.startswith(crypto.MAGIC)
    assert crypto.decrypt_bytes(blob, dek) == b"hello world"


def test_wrong_key_fails():
    blob = crypto.encrypt_bytes(b"secret", b"\x01" * 32)
    with pytest.raises(crypto.AuthError):
        crypto.decrypt_bytes(blob, b"\x02" * 32)


def test_is_encrypted_on_plaintext():
    assert not crypto.is_encrypted(b'{"a": 1}')


# ------------------------------ keyfile / login ------------------------------
def test_keyfile_create_and_unlock(tmp_path):
    kf = tmp_path / "app_key.enc"
    dek = crypto.create_keyfile(kf, password="admin", username="admin")
    assert crypto.keyfile_exists(kf)
    assert crypto.unlock(kf, "admin") == dek           # right password -> same DEK
    assert crypto.status(kf)["using_default_creds"] is True


def test_unlock_wrong_password(tmp_path):
    kf = tmp_path / "app_key.enc"
    crypto.create_keyfile(kf, password="admin")
    with pytest.raises(crypto.AuthError):
        crypto.unlock(kf, "wrong")


def test_change_password_rewraps_same_dek(tmp_path):
    kf = tmp_path / "app_key.enc"
    dek = crypto.create_keyfile(kf, password="admin", username="admin")
    crypto.change_credentials(kf, old_password="admin",
                              new_username="alice", new_password="s3cret!")
    # Old password no longer works; new one unlocks the *same* DEK (no re-encryption).
    with pytest.raises(crypto.AuthError):
        crypto.unlock(kf, "admin")
    assert crypto.unlock(kf, "s3cret!") == dek
    st = crypto.status(kf)
    assert st["username"] == "alice"
    assert st["using_default_creds"] is False


def test_dismiss_warning(tmp_path):
    kf = tmp_path / "app_key.enc"
    crypto.create_keyfile(kf, password="admin")
    assert crypto.status(kf)["warning_dismissed"] is False
    crypto.dismiss_default_warning(kf)
    assert crypto.status(kf)["warning_dismissed"] is True


# ------------------------------ core file funnel ------------------------------
def test_save_json_encrypts_at_rest(tmp_path):
    crypto.set_active_key(b"\x07" * 32)
    p = tmp_path / "chats.json"
    core.save_json(p, [{"id": "x", "title": "hi"}])
    raw = p.read_bytes()
    assert crypto.is_encrypted(raw)                     # ciphertext on disk
    assert b"title" not in raw                          # no plaintext leaks
    assert core.load_json(p, None) == [{"id": "x", "title": "hi"}]  # transparent read-back


def test_load_json_reads_legacy_plaintext(tmp_path):
    # A pre-migration plaintext file is still readable once unlocked.
    p = tmp_path / "presets.json"
    p.write_text('[{"name": "a"}]', encoding="utf-8")
    crypto.set_active_key(b"\x09" * 32)
    assert core.load_json(p, None) == [{"name": "a"}]


def test_plain_helpers_never_encrypt(tmp_path):
    crypto.set_active_key(b"\x03" * 32)                 # even when unlocked...
    p = tmp_path / "profiles.json"
    core.save_json_plain(p, {"active": "default"})
    raw = p.read_bytes()
    assert not crypto.is_encrypted(raw)                 # ...registries stay plaintext
    assert core.load_json_plain(p, None) == {"active": "default"}
