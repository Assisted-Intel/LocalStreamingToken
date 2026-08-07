#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Profile management. The app has two independent axes of profiles:

  * DATA profiles     — one folder per profile under ``data/profiles/<id>/`` holding
                        chats, prompts, resources (libraries), prompt-eval projects,
                        database sessions + their encrypted credential vault, and the
                        RAG vector store. Includes a private **Incognito** session that
                        lives in a scratch folder and is never carried across restarts.
  * SETTINGS profiles — one folder per profile under ``settings/profiles/<id>/`` holding
                        ``settings.json`` (providers, API keys, web-search tokens,
                        RAG-embed + parallel config, general defaults).

Switching one axis never touches the other. ``ProfileManager`` owns the two registry
files (``data/profiles.json`` and ``settings/profiles.json``) and the active ids; the
actual runtime swap (repointing core paths, reloading the Store, re-locking the vault,
resetting DuckDB connections) is orchestrated by the server.
"""

import shutil
import uuid
from pathlib import Path

from . import core
# The profile REGISTRIES stay plaintext: the ProfileManager reads them at boot, before
# the user logs in and the encryption key is available. (The per-profile data files
# they point to ARE encrypted.)
from .core import load_json_plain as load_json, save_json_plain as save_json

DEFAULT_ID = "default"
DEFAULT_NAME = "Default"

# Files copied when a new DATA profile is created by "clone from current" (prompt
# libraries + resources + presets — deliberately NOT chat history / evals / db).
CLONE_FILE_NAMES = ("prompts.json", "libraries.json", "presets.json")


def _new_id():
    return uuid.uuid4().hex[:12]


def _move(src: Path, dst: Path):
    """Move src -> dst if src exists and dst doesn't. Best effort."""
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def migrate_if_needed():
    """First-run upgrade: fold pre-profile flat files into a single ``Default`` profile
    on each axis. Idempotent — does nothing once the registries exist."""
    # ---- DATA axis ----
    if not core.DATA_REGISTRY_FILE.exists():
        dest = core.DATA_PROFILES_DIR / DEFAULT_ID
        dest.mkdir(parents=True, exist_ok=True)
        for name in core.DATA_FILE_NAMES:
            _move(core.DATA_DIR / name, dest / name)
        # RAG store (+ write-ahead log) and the DB staging/audit tree.
        _move(core.DATA_DIR / "rag.duckdb", dest / "rag.duckdb")
        _move(core.DATA_DIR / "rag.duckdb.wal", dest / "rag.duckdb.wal")
        _move(core.DATA_DIR / "db", dest / "db")
        # The credential vault used to live in settings/; it now belongs to the profile.
        _move(core.SETTINGS_DIR / "db_vault.enc", dest / "db_vault.enc")
        save_json(core.DATA_REGISTRY_FILE,
                  {"active": DEFAULT_ID, "profiles": [{"id": DEFAULT_ID, "name": DEFAULT_NAME}]})

    # ---- SETTINGS axis ----
    if not core.SETTINGS_REGISTRY_FILE.exists():
        dest = core.SETTINGS_PROFILES_DIR / DEFAULT_ID
        dest.mkdir(parents=True, exist_ok=True)
        _move(core.SETTINGS_DIR / "settings.json", dest / "settings.json")
        save_json(core.SETTINGS_REGISTRY_FILE,
                  {"active": DEFAULT_ID, "profiles": [{"id": DEFAULT_ID, "name": DEFAULT_NAME}]})


class ProfileManager:
    def __init__(self):
        migrate_if_needed()
        self.data_reg = self._load_registry(core.DATA_REGISTRY_FILE)
        self.settings_reg = self._load_registry(core.SETTINGS_REGISTRY_FILE)
        self._incognito = False
        # Wipe any leftover incognito scratch from a previous (possibly crashed) run.
        self._wipe_incognito_scratch()

    # ------------------------------ registries ------------------------------
    @staticmethod
    def _load_registry(path):
        reg = load_json(path, None)
        if not isinstance(reg, dict) or not reg.get("profiles"):
            reg = {"active": DEFAULT_ID, "profiles": [{"id": DEFAULT_ID, "name": DEFAULT_NAME}]}
            save_json(path, reg)
        # Make sure active points at a real profile.
        ids = {p["id"] for p in reg["profiles"]}
        if reg.get("active") not in ids:
            reg["active"] = reg["profiles"][0]["id"]
        return reg

    def _save_data(self):
        save_json(core.DATA_REGISTRY_FILE, self.data_reg)

    def _save_settings(self):
        save_json(core.SETTINGS_REGISTRY_FILE, self.settings_reg)

    # ------------------------------ paths ------------------------------
    def data_dir(self, profile_id):
        return core.DATA_PROFILES_DIR / profile_id

    def settings_dir(self, profile_id):
        return core.SETTINGS_PROFILES_DIR / profile_id

    @staticmethod
    def _deletable_dir(root: Path, profile_id):
        """The folder to ``rmtree`` for ``profile_id``, or raise.

        Profile ids come straight off the URL path, and ``root / profile_id`` happily
        accepts a relative segment: ``".."`` resolves to ``data/`` or ``settings/`` —
        the latter holding ``app_key.enc``, without which every encrypted file is
        unrecoverable. So the id must be a single plain segment AND the resolved path
        must still sit directly under ``root``. Callers additionally check the id is a
        registered profile; this is the second belt."""
        pid = str(profile_id or "")
        if not pid or pid in (".", "..") or "/" in pid or "\\" in pid:
            raise ValueError("Invalid profile id.")
        target = (root / pid).resolve()
        if target.parent != Path(root).resolve():
            raise ValueError("Invalid profile id.")
        return target

    def active_data_id(self):
        return self.data_reg["active"]

    def active_settings_id(self):
        return self.settings_reg["active"]

    def active_data_dir(self):
        """Where data reads/writes go right now — the incognito scratch when private."""
        if self._incognito:
            return core.INCOGNITO_DIR
        return self.data_dir(self.active_data_id())

    def active_settings_dir(self):
        return self.settings_dir(self.active_settings_id())

    # ------------------------------ state (for the client) ------------------------------
    def state(self):
        return {
            "data": {
                "active": None if self._incognito else self.active_data_id(),
                "incognito": self._incognito,
                "profiles": [dict(p) for p in self.data_reg["profiles"]],
            },
            "settings": {
                "active": self.active_settings_id(),
                "profiles": [dict(p) for p in self.settings_reg["profiles"]],
            },
        }

    # ------------------------------ create / rename / delete ------------------------------
    def create_data(self, name, seed="blank"):
        """Create a new data profile folder. seed='clone' copies the CURRENT profile's
        prompt library, resources and presets (not chats). Returns the profile dict."""
        pid = _new_id()
        dest = self.data_dir(pid)
        dest.mkdir(parents=True, exist_ok=True)
        if seed == "clone" and not self._incognito:
            src = self.active_data_dir()
            for fname in CLONE_FILE_NAMES:
                if (src / fname).exists():
                    shutil.copy2(str(src / fname), str(dest / fname))
        prof = {"id": pid, "name": (name or "Untitled").strip() or "Untitled"}
        self.data_reg["profiles"].append(prof)
        self._save_data()
        return prof

    def create_settings(self, name):
        pid = _new_id()
        self.settings_dir(pid).mkdir(parents=True, exist_ok=True)
        prof = {"id": pid, "name": (name or "Untitled").strip() or "Untitled"}
        self.settings_reg["profiles"].append(prof)
        self._save_settings()
        return prof

    def rename_data(self, profile_id, name):
        for p in self.data_reg["profiles"]:
            if p["id"] == profile_id:
                p["name"] = (name or "").strip() or p["name"]
                self._save_data()
                return dict(p)
        return None

    def rename_settings(self, profile_id, name):
        for p in self.settings_reg["profiles"]:
            if p["id"] == profile_id:
                p["name"] = (name or "").strip() or p["name"]
                self._save_settings()
                return dict(p)
        return None

    def delete_data(self, profile_id):
        """Delete a data profile and its folder. Can't delete the active one or the last."""
        if profile_id not in {p["id"] for p in self.data_reg["profiles"]}:
            raise ValueError("No such data profile.")
        if self._incognito or profile_id == self.active_data_id():
            raise ValueError("Switch to another profile before deleting this one.")
        if len(self.data_reg["profiles"]) <= 1:
            raise ValueError("You can't delete the last profile.")
        # Resolve (and validate) the folder BEFORE dropping the registry entry, so a
        # rejected id leaves the registry untouched.
        target = self._deletable_dir(core.DATA_PROFILES_DIR, profile_id)
        self.data_reg["profiles"] = [p for p in self.data_reg["profiles"] if p["id"] != profile_id]
        self._save_data()
        shutil.rmtree(target, ignore_errors=True)
        return True

    def delete_settings(self, profile_id):
        if profile_id not in {p["id"] for p in self.settings_reg["profiles"]}:
            raise ValueError("No such settings profile.")
        if profile_id == self.active_settings_id():
            raise ValueError("Switch to another settings profile before deleting this one.")
        if len(self.settings_reg["profiles"]) <= 1:
            raise ValueError("You can't delete the last settings profile.")
        target = self._deletable_dir(core.SETTINGS_PROFILES_DIR, profile_id)
        self.settings_reg["profiles"] = [p for p in self.settings_reg["profiles"] if p["id"] != profile_id]
        self._save_settings()
        shutil.rmtree(target, ignore_errors=True)
        return True

    # ------------------------------ activation ------------------------------
    def set_active_data(self, profile_id):
        if profile_id not in {p["id"] for p in self.data_reg["profiles"]}:
            raise ValueError("No such data profile.")
        self._incognito = False
        self.data_reg["active"] = profile_id
        self._save_data()

    def set_active_settings(self, profile_id):
        if profile_id not in {p["id"] for p in self.settings_reg["profiles"]}:
            raise ValueError("No such settings profile.")
        self.settings_reg["active"] = profile_id
        self._save_settings()

    # ------------------------------ incognito ------------------------------
    @property
    def incognito(self):
        return self._incognito

    def _wipe_incognito_scratch(self):
        shutil.rmtree(core.INCOGNITO_DIR, ignore_errors=True)

    def start_incognito(self, seed="blank"):
        """Enter a private session. Nothing is written to a real profile until the user
        saves. seed='clone' pre-loads a working copy of the current profile's prompt
        library, resources and presets so the session isn't empty."""
        prev_dir = self.active_data_dir()
        self._wipe_incognito_scratch()
        core.INCOGNITO_DIR.mkdir(parents=True, exist_ok=True)
        if seed == "clone":
            for fname in CLONE_FILE_NAMES:
                if (prev_dir / fname).exists():
                    shutil.copy2(str(prev_dir / fname), str(core.INCOGNITO_DIR / fname))
        self._incognito = True

    def stop_incognito(self):
        """Leave incognito WITHOUT saving; drop back to the last real active profile."""
        self._incognito = False
        self._wipe_incognito_scratch()
