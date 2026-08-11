#!/usr/bin/env python3
"""The one-time encryption sweep (app/migrate.py).

The headline guarantee — that the sweep never touches a byte inside a ``*.lance``
directory — is already pinned by ``test_encryption_sweep_leaves_the_lance_store_alone``
in tests/test_vectorstore.py, alongside the ``wipe_encrypted`` counterpart. That test
exists because the sweep once destroyed a real store: every internal file grew by the
36-byte magic+nonce+tag and Lance could only report
``LanceError(IO): file size is too small``.

This file covers the properties that test doesn't, which matter because ``run()`` is
invoked on **every login** rather than once:

* it is idempotent, so repeated logins never double-encrypt;
* it is a no-op while locked, so a pre-login walk can't rewrite files with no key;
* the carve-out is asserted per *file kind*, so a regression names the shape it missed
  (``latest_version_hint.json`` is the trap — it ends in .json like the app's own files);
* the plaintext profile registries survive both the sweep and the wipe.
"""

import hashlib

import pytest

from app import crypto, migrate
from conftest import isolate_paths


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """An unlocked, isolated data/settings tree."""
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    yield tmp_path
    crypto.clear_key()


def _lance_store(tmp_path):
    """A directory shaped like a real Lance store. The nesting is the point:
    ``rag.lance/chunks_768.lance/data/*.lance`` is what a check looking only at the
    parent directory, or only at the suffix, would wave through."""
    table = tmp_path / "data" / "profiles" / "default" / "rag.lance" / "chunks_768.lance"
    for rel in ("data/0101101011000101.lance",
                "_versions/18446744073709551601.manifest",
                "_versions/latest_version_hint.json",
                "_transactions/0-43b7006e-dfb0.txn",
                "_indices/76b45301-c451/index.idx",
                "_indices/76b45301-c451/auxiliary.idx"):
        p = table / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"LANCE-INTERNAL-BINARY-PAYLOAD" * 4)
    return table.parent


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("relative", [
    "chunks_768.lance/data/0101101011000101.lance",
    "chunks_768.lance/_versions/18446744073709551601.manifest",
    "chunks_768.lance/_versions/latest_version_hint.json",
    "chunks_768.lance/_transactions/0-43b7006e-dfb0.txn",
    "chunks_768.lance/_indices/76b45301-c451/index.idx",
    "chunks_768.lance/_indices/76b45301-c451/auxiliary.idx",
])
def test_should_skip_covers_every_kind_of_lance_internal(profile, relative):
    """One file kind per case, so a regression reports which shape slipped through
    instead of one opaque byte-comparison failure."""
    assert migrate._should_skip(_lance_store(profile) / relative) is True


def test_the_sweep_is_idempotent(profile):
    """It runs on every login, so a second pass must re-encrypt nothing — otherwise
    files would be wrapped again and again until they no longer decrypt."""
    root = _lance_store(profile)
    prof = profile / "data" / "profiles" / "default"
    (prof / "libraries.json").write_bytes(b'{"libraries": []}')

    first = migrate.run()
    assert first["encrypted"] >= 1
    after_first = _hashes(profile / "data")

    second = migrate.run()
    assert second["encrypted"] == 0
    assert _hashes(profile / "data") == after_first
    assert _hashes(root)                      # store still present, still untouched


def test_a_locked_app_sweeps_nothing(profile):
    """No DEK, no sweep. core.write_bytes silently writes PLAINTEXT when locked, so a
    sweep in that state would rewrite every file while marking none of them encrypted."""
    root = _lance_store(profile)
    prof = profile / "data" / "profiles" / "default"
    (prof / "libraries.json").write_bytes(b'{"libraries": []}')
    before = _hashes(profile / "data")
    crypto.clear_key()

    assert migrate.run() == {"skipped": "locked"}
    assert _hashes(profile / "data") == before
    assert _hashes(root)


def test_the_profile_registry_survives_both_the_sweep_and_the_wipe(profile):
    """profiles.json must stay readable BEFORE login — the ProfileManager reads it at
    boot to find the active profile, long before any key exists."""
    _lance_store(profile)
    reg = profile / "data" / "profiles.json"
    reg.write_bytes(b'{"profiles": []}')

    migrate.run()
    assert not crypto.is_encrypted(reg.read_bytes())

    migrate.wipe_encrypted()
    assert reg.exists()
