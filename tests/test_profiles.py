#!/usr/bin/env python3
"""Tests for profile management (app/profiles.py).

The focus is deletion. A profile id arrives straight off the URL path
(``DELETE /api/profiles/data/<profile_id>``) and used to be joined onto the profiles
root and handed to ``shutil.rmtree`` with no validation at all — so ``..`` resolved to
``data/`` or, worse, ``settings/``, which holds ``app_key.enc``. Losing that keyfile
makes every encrypted file in the install permanently unreadable; there is no second
copy of the key. These tests pin the guard in place.

Everything runs against a throwaway tree; no real profile is touched.
"""

import pytest

from app import core, profiles


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A complete, isolated data+settings tree with a ProfileManager pointed at it."""
    data, settings = tmp_path / "data", tmp_path / "settings"
    monkeypatch.setattr(core, "DATA_DIR", data)
    monkeypatch.setattr(core, "SETTINGS_DIR", settings)
    monkeypatch.setattr(core, "DATA_PROFILES_DIR", data / "profiles")
    monkeypatch.setattr(core, "SETTINGS_PROFILES_DIR", settings / "profiles")
    monkeypatch.setattr(core, "DATA_REGISTRY_FILE", data / "profiles.json")
    monkeypatch.setattr(core, "SETTINGS_REGISTRY_FILE", settings / "profiles.json")
    monkeypatch.setattr(core, "INCOGNITO_DIR", data / "profiles" / ".incognito")
    for d in (data, settings, core.DATA_PROFILES_DIR, core.SETTINGS_PROFILES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # The two things that must never be collateral damage.
    (settings / "app_key.enc").write_text("WRAPPED-DEK")
    (data / "chats-of-another-profile.json").write_text("[]")
    pm = profiles.ProfileManager()
    return pm, tmp_path


# Single path segments that are NOT profile ids. "." and ".." are the dangerous ones;
# the rest simply must not be treated as deletable.
BAD_IDS = ["..", ".", "", "../..", "unknown-id", "a/b", "a\\b"]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_delete_data_rejects_ids_that_are_not_profiles(tree, bad):
    pm, root = tree
    pm.create_data("Second")
    with pytest.raises(ValueError):
        pm.delete_data(bad)
    assert (root / "data").is_dir()
    assert (root / "data" / "profiles").is_dir()
    assert (root / "data" / "chats-of-another-profile.json").is_file()


@pytest.mark.parametrize("bad", BAD_IDS)
def test_delete_settings_rejects_ids_that_are_not_profiles(tree, bad):
    pm, root = tree
    pm.create_settings("Second")
    with pytest.raises(ValueError):
        pm.delete_settings(bad)
    assert (root / "settings").is_dir()
    assert (root / "settings" / "profiles").is_dir()
    # The keyfile: losing this is unrecoverable, not merely inconvenient.
    assert (root / "settings" / "app_key.enc").read_text() == "WRAPPED-DEK"


def test_deleting_a_real_profile_still_works(tree):
    pm, _root = tree
    prof = pm.create_data("Second")
    folder = core.DATA_PROFILES_DIR / prof["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chats.json").write_text("[]")
    assert pm.delete_data(prof["id"]) is True
    assert not folder.exists()
    assert prof["id"] not in {p["id"] for p in pm.data_reg["profiles"]}

    sprof = pm.create_settings("Second")
    sfolder = core.SETTINGS_PROFILES_DIR / sprof["id"]
    sfolder.mkdir(parents=True, exist_ok=True)
    assert pm.delete_settings(sprof["id"]) is True
    assert not sfolder.exists()


def test_cannot_delete_the_active_or_the_last_profile(tree):
    pm, _root = tree
    pm.create_data("Second")
    with pytest.raises(ValueError):
        pm.delete_data(pm.active_data_id())
    # Back down to one profile: the last one is not deletable either.
    others = [p["id"] for p in pm.data_reg["profiles"] if p["id"] != pm.active_data_id()]
    for pid in others:
        pm.delete_data(pid)
    with pytest.raises(ValueError):
        pm.delete_data(pm.active_data_id())


def test_a_rejected_delete_leaves_the_registry_untouched(tree):
    """The folder is resolved before the registry entry is dropped, so a refusal can't
    leave a registered profile pointing at nothing."""
    pm, _root = tree
    pm.create_data("Second")
    before = [dict(p) for p in pm.data_reg["profiles"]]
    with pytest.raises(ValueError):
        pm.delete_data("..")
    assert pm.data_reg["profiles"] == before
