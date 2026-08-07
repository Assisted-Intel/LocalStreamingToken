#!/usr/bin/env python3
"""Shared test harness.

Every test file used to carry its own verbatim copy of the eight ``core.*`` path
monkeypatches plus a logged-in Flask client, and two files hand-rolled their own
(subtly different) SSE parsers. This is that harness, once.

Three things live here:

* ``isolate_paths`` / the ``store`` and ``client`` fixtures — point the app's data and
  settings at a throwaway tree so no real profile is ever touched;
* ``sse_frames`` — one SSE parser;
* ``StubAdapter`` / ``use_adapter`` — the seam for routes that call a model.

Domain-specific fakes stay in the file that needs them (``FakeAdapter`` in
test_evals_routes.py routes on a grader marker, which only that suite cares about).
"""

import json

import pytest

from app import core, crypto, profiles, providers


# --------------------------- isolation ---------------------------

def isolate_paths(tmp_path, monkeypatch, unlock=False):
    """Redirect every data/settings path at ``tmp_path`` and create the keyfile.

    Must run BEFORE ``server.create_app()`` — ``Store()`` reads ``core.*`` at
    construction, which is why every caller imports ``app.server`` lazily afterwards.

    ``unlock`` activates the data encryption key directly, for tests that build a
    ``Store`` without going through ``/api/login`` (the route is what normally
    activates it).
    """
    data, settings = tmp_path / "data", tmp_path / "settings"
    monkeypatch.setattr(core, "DATA_DIR", data)
    monkeypatch.setattr(core, "SETTINGS_DIR", settings)
    monkeypatch.setattr(core, "DATA_PROFILES_DIR", data / "profiles")
    monkeypatch.setattr(core, "SETTINGS_PROFILES_DIR", settings / "profiles")
    monkeypatch.setattr(core, "DATA_REGISTRY_FILE", data / "profiles.json")
    monkeypatch.setattr(core, "SETTINGS_REGISTRY_FILE", settings / "profiles.json")
    monkeypatch.setattr(core, "INCOGNITO_DIR", data / "profiles" / ".incognito")
    monkeypatch.setattr(core, "APP_KEYFILE", settings / "app_key.enc")
    for d in (data, settings):
        d.mkdir(parents=True, exist_ok=True)
    crypto.create_keyfile(core.APP_KEYFILE)          # default admin/admin
    if unlock:
        # The store's data files are encrypted at rest and unreadable until the DEK is
        # active — the same gate /api/login passes through.
        crypto.set_active_key(crypto.unlock(core.APP_KEYFILE, "admin"))

    pm = profiles.ProfileManager()
    core.set_active_settings_profile(pm.active_settings_dir())
    core.set_active_data_profile(pm.active_data_dir())


def make_client(tmp_path, monkeypatch):
    """Build and log into a test client. Split out from the fixture so a suite needing
    extra setup or teardown (test_db_routes releases its staging file) can reuse it."""
    isolate_paths(tmp_path, monkeypatch)
    from app import server
    app = server.create_app()
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200, r.get_data(as_text=True)
    return c


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real Store whose data lives entirely under tmp_path."""
    from app import store as store_mod
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    return store_mod.Store()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A logged-in test client for an app whose data lives entirely under tmp_path.
    No model server is reachable, which is exactly the state these routes must behave
    sanely in — install a stub with ``use_adapter`` when one is needed."""
    return make_client(tmp_path, monkeypatch)


# --------------------------- SSE ---------------------------

def sse_frames(resp):
    """Parse an SSE response body into ``[(event, data-dict), ...]``.

    The body HAS to be consumed for the route to do anything: ``stream_with_context``
    is lazy, so the handler's generator does not run until something reads it. Calling
    this is what runs the route.
    """
    out = []
    for block in resp.get_data(as_text=True).split("\n\n"):
        event, data = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event and data:
            out.append((event, json.loads(data)))
    return out


def events(frames):
    return [e for e, _ in frames]


def first(frames, name):
    return next(d for e, d in frames if e == name)


def all_of(frames, name):
    return [d for e, d in frames if e == name]


# --------------------------- the model seam ---------------------------

class StubAdapter:
    """A provider adapter replaying canned answers, in call order.

    Deliberately has NO ``client`` attribute. ``rewrite.run_completion`` short-circuits
    to ``client.complete`` whenever ``fmt`` is set and the adapter exposes one
    (rewrite.py:51) — and every memory and persona pass sends a schema — so an adapter
    with a mock client would never reach ``chat_stream`` and the stub would be silently
    bypassed. Do not add one.

    A reply may be an ``Exception`` instance, which is raised instead of yielded; that
    is how a failing model call is injected. It may also be a *list* of ``(kind,
    payload)`` frames, which are yielded verbatim — the way an ``("image", ...)`` or
    ``("reasoning", ...)`` stream is injected.
    """

    def __init__(self, replies=(), models=("test-model",)):
        self.replies = list(replies)
        self._models = list(models)
        self.seen = []          # [(model, messages, options)] for every call

    # --- assertion helpers ---
    @property
    def calls(self):
        return len(self.seen)

    def prompt(self, i=0):
        """The i-th call's messages flattened to one string."""
        return "\n".join(m.get("content", "") for m in self.seen[i][1])

    def system(self, i=0):
        for m in self.seen[i][1]:
            if m.get("role") == "system":
                return m.get("content", "")
        return ""

    # --- adapter interface ---
    def list_models(self):
        return list(self._models)

    def model_capabilities(self, model):
        return []               # not a reasoning model, no tool support

    def chat_stream(self, model, messages, options, stop_event, **kw):
        self.seen.append((model, messages, options))
        reply = self.replies.pop(0) if self.replies else ""
        if callable(reply):
            reply = reply(self)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            yield from reply
            return
        yield ("content", reply)


def use_adapter(monkeypatch, adapter):
    """Point every provider lookup at one adapter.

    ``adapter_for`` is a closure inside ``create_app``, so the seam is
    ``providers.get_client`` — which also covers the one place (``_ollama_caps``) that
    calls it directly rather than through ``adapter_for``.
    """
    monkeypatch.setattr(providers, "get_client", lambda server: adapter)
    return adapter
