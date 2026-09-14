#!/usr/bin/env python3
"""Avatar Read Server launcher (/api/avatar/ensure)."""

import pytest


@pytest.fixture(autouse=True)
def reset_owned_helper():
    from app import avatar
    avatar._started = None
    yield
    avatar._started = None



def test_voice_status_reads_the_helper_ready_file(client, tmp_path):
    status = tmp_path / "voice-status.json"
    status.write_text('{"ready": true, "detail": "Ready. TTS and transcription are loaded.", "stt": "ready", "tts": "ready"}', encoding="utf-8")
    client.post("/api/settings", json={"avatar_dir": str(tmp_path)})
    body = client.get("/api/avatar/voice-status").get_json()
    assert body["ready"] is True
    assert "Ready" in (body.get("detail") or "")


def test_voice_status_missing_file_is_not_ready(client, tmp_path):
    client.post("/api/settings", json={"avatar_dir": str(tmp_path)})
    body = client.get("/api/avatar/voice-status").get_json()
    assert body["ready"] is False
    assert body.get("missing") is True


def test_ensure_reports_already_running(client, monkeypatch):
    from app import avatar

    monkeypatch.setattr(avatar, "health", lambda url, timeout=1.2: True)
    monkeypatch.setattr(avatar, "spawn", lambda config: (_ for _ in ()).throw(
        AssertionError("must not spawn when /health already answers")))
    body = client.post("/api/avatar/ensure").get_json()
    assert body["ok"] is True
    assert body["running"] is True
    assert body["started"] is False
    assert "8765" in body["url"]


def test_ensure_does_not_spawn_a_second_helper_when_ours_is_still_alive(client, monkeypatch):
    from app import avatar
    from types import SimpleNamespace

    spawned = {"n": 0}
    monkeypatch.setattr(avatar, "health", lambda url, timeout=1.2: False)
    monkeypatch.setattr(avatar, "_port_open", lambda url: False)
    monkeypatch.setattr(avatar, "spawn", lambda config: spawned.__setitem__("n", spawned["n"] + 1))
    monkeypatch.setattr(avatar, "_started", SimpleNamespace(poll=lambda: None))
    orig = avatar.ensure_started
    monkeypatch.setattr(avatar, "ensure_started", lambda config, wait=30: orig(config, wait=0.15))
    # health stays false; owned child is alive — must not launch another python.
    body = client.post("/api/avatar/ensure").get_json()
    assert spawned["n"] == 0
    assert body["running"] is True


def test_ensure_does_not_spawn_when_the_port_is_already_taken(client, monkeypatch):
    from app import avatar

    spawned = {"n": 0}
    monkeypatch.setattr(avatar, "health", lambda url, timeout=1.2: False)
    monkeypatch.setattr(avatar, "_port_open", lambda url: True)
    monkeypatch.setattr(avatar, "spawn", lambda config: spawned.__setitem__("n", spawned["n"] + 1))
    orig = avatar.ensure_started
    monkeypatch.setattr(avatar, "ensure_started", lambda config, wait=30: orig(config, wait=0.15))
    body = client.post("/api/avatar/ensure").get_json()
    assert spawned["n"] == 0
    assert body["ok"] is True
    assert body["started"] is False


def test_ensure_without_a_folder_fails_instead_of_spawning(client, monkeypatch):
    from app import avatar

    monkeypatch.setattr(avatar, "health", lambda url, timeout=1.2: False)
    monkeypatch.setattr(avatar, "_port_open", lambda url: False)
    r = client.post("/api/avatar/ensure")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert body["running"] is False
    assert "folder" in (body.get("error") or "").lower()


def test_spawn_uses_this_interpreter(tmp_path, monkeypatch):
    """The helper must start with the same python that is running Local Streaming Token
    (the F5 conda env), not a second executable from Settings."""
    import sys
    from app import avatar

    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "app.py").write_text("# stub\n", encoding="utf-8")
    captured = {}

    class FakePopen:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(avatar.subprocess, "Popen", FakePopen)
    avatar.spawn({"avatar_dir": str(helper)})
    assert captured["args"][0] == sys.executable
    assert captured["args"][1:] == ["app.py", "--no-browser"]
    assert captured["cwd"] == str(helper)


def test_health_rejects_html_so_this_app_is_not_mistaken_for_the_helper(monkeypatch):
    from app import avatar

    class FakeResp:
        status = 200
        def read(self):
            return b"<!doctype html><title>Local Streaming Token</title>"
        def getcode(self):
            return 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class FakeOpener:
        def open(self, req, timeout=None):
            return FakeResp()

    monkeypatch.setattr(avatar, "_OPENER", FakeOpener())
    assert avatar.health("http://127.0.0.1:8765") is False


def test_rpc_rejects_unknown_paths(client):
    r = client.post("/api/avatar/rpc", json={"path": "/not-a-thing", "method": "GET"})
    assert r.status_code == 400
    assert "not allowed" in r.get_json()["error"].lower()


def test_rpc_forwards_status(client, monkeypatch):
    from app import avatar

    monkeypatch.setattr(avatar, "rpc", lambda config, path, method="GET", body=None, timeout=20: {
        "ok": True, "path": path, "listen": {"state": "listening"},
    })
    body = client.post("/api/avatar/rpc", json={"path": "/status", "method": "GET"}).get_json()
    assert body["ok"] is True
    assert body["listen"]["state"] == "listening"


def test_stop_owned_only_kills_a_helper_this_process_started(monkeypatch):
    from app import avatar

    class FakeProc:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self._alive = True
        def poll(self):
            return None if self._alive else 0
        def terminate(self):
            self.terminated = True
            self._alive = False
        def wait(self, timeout=None):
            return 0
        def kill(self):
            self.killed = True
            self._alive = False

    monkeypatch.setattr(avatar, "_started", None)
    assert avatar.stop_owned() is False

    owned = FakeProc()
    monkeypatch.setattr(avatar, "_started", owned)
    assert avatar.stop_owned() is True
    assert owned.terminated is True
    assert avatar._started is None
    # Second call is a no-op — we must not hunt down a helper we did not spawn.
    assert avatar.stop_owned() is False


def test_ensure_does_not_claim_ownership_when_helper_already_running(client, monkeypatch):
    from app import avatar

    monkeypatch.setattr(avatar, "_started", None)
    monkeypatch.setattr(avatar, "health", lambda url, timeout=1.2: True)
    monkeypatch.setattr(avatar, "spawn", lambda config: (_ for _ in ()).throw(
        AssertionError("must not spawn")))
    client.post("/api/avatar/ensure")
    assert avatar._started is None


def test_restart_stops_owned_helper_then_starts_again(client, monkeypatch):
    from app import avatar
    from types import SimpleNamespace

    stopped = {"n": 0}
    monkeypatch.setattr(avatar, "stop_owned", lambda: stopped.__setitem__("n", stopped["n"] + 1) or True)
    monkeypatch.setattr(avatar, "_wait_until_down", lambda url, wait=10: True)
    monkeypatch.setattr(avatar, "ensure_started", lambda config, wait=30: {
        "ok": True, "running": True, "started": True, "url": "http://127.0.0.1:8765",
    })
    monkeypatch.setattr(avatar, "_started", SimpleNamespace(poll=lambda: None))
    body = client.post("/api/avatar/restart").get_json()
    assert body["ok"] is True
    assert body["started"] is True
    assert stopped["n"] == 1


def test_ensure_spawns_and_waits_for_health(client, monkeypatch):
    from app import avatar
    from types import SimpleNamespace

    hits = {"n": 0}

    def health(url, timeout=1.2):
        hits["n"] += 1
        return hits["n"] >= 4

    monkeypatch.setattr(avatar, "health", health)
    monkeypatch.setattr(avatar, "_port_open", lambda url: False)
    monkeypatch.setattr(avatar, "spawn", lambda config: SimpleNamespace(poll=lambda: None))
    body = client.post("/api/avatar/ensure").get_json()
    assert body["ok"] is True
    assert body["started"] is True
    assert body["running"] is True
