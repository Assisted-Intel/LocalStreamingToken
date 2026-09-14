"""Start and probe the Avatar Read Server used for voice chat.

The helper is a separate process. It is spawned with *this* interpreter
(``sys.executable``), because Local Streaming Token is already running in the
F5 conda env that has Parakeet and F5. ensure_started() health-checks the
configured URL and, if needed, runs ``app.py --no-browser`` in the helper
folder.

If this process spawned the helper, Ctrl+C / process exit stops it. A helper
that was already running (or started by something else) is left alone.
"""

from __future__ import annotations

import atexit
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_URL = "http://127.0.0.1:8765"
HEALTH_TIMEOUT = 1.2
SPAWN_WAIT = 30.0
RPC_ALLOWED = {
    "/status": "GET",
    "/health": "GET",
    "/ready": "GET",
    "/speak": "POST",
    "/stop": "POST",
    "/settings": "POST",
    "/listen/start": "POST",
    "/listen/cancel": "POST",
    "/listen/ack": "POST",
    "/shutdown": "POST",
}

_started = None  # Popen of a helper this process spawned, or None
_spawn_lock = threading.Lock()
# Windows HTTP_PROXY must not intercept 127.0.0.1 or health looks up while rpc dies.
_OPENER = build_opener(ProxyHandler({}))


def helper_url(config) -> str:
    raw = str((config or {}).get("avatar_url") or DEFAULT_URL).strip() or DEFAULT_URL
    return raw.rstrip("/")


def _request(url, method="GET", body=None, timeout=HEALTH_TIMEOUT):
    data = None
    headers = {}
    if method != "GET" and body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8") or "{}"
        code = getattr(resp, "status", None) or resp.getcode()
        return int(code), raw


def health(url: str, timeout: float = HEALTH_TIMEOUT) -> bool:
    """True only if the Avatar helper's /health JSON answers. A 200 HTML page
    (this app's SPA fallback) must not count as the helper."""
    target = helper_url({"avatar_url": url}) + "/health"
    try:
        code, raw = _request(target, timeout=timeout)
        if not (200 <= code < 300):
            return False
        data = json.loads(raw)
        return bool(data.get("ok")) and ("tts_engine" in data or "voice" in data)
    except (URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def _port_open(url: str) -> bool:
    """True if something is already listening on the helper host:port."""
    parsed = urlparse(helper_url({"avatar_url": url}))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _windows_flags():
    if sys.platform != "win32":
        return 0
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — lives past a Flask reload.
    return 0x00000008 | 0x00000200


def _validate(config) -> Path:
    folder = Path(str((config or {}).get("avatar_dir") or "").strip())
    if not folder or not str(folder):
        raise ValueError("Set the Avatar helper folder in Settings.")
    if not folder.is_dir():
        raise ValueError(f"Avatar helper folder is missing: {folder}")
    app_py = folder / "app.py"
    if not app_py.is_file():
        raise ValueError(f"No app.py in the Avatar helper folder: {folder}")
    return folder


def spawn(config) -> subprocess.Popen:
    folder = _validate(config)
    log_path = folder / "lst-spawn.log"
    log_handle = open(log_path, "ab", buffering=0)
    popen_kw = {
        "args": [sys.executable, "app.py", "--no-browser"],
        "cwd": str(folder),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = _windows_flags()
        popen_kw["close_fds"] = False  # needed to keep the log handle
    else:
        popen_kw["start_new_session"] = True
    proc = subprocess.Popen(**popen_kw)
    # The child owns the log fd; we can close our copy.
    try:
        log_handle.close()
    except Exception:
        pass
    return proc


def ensure_started(config, wait: float = SPAWN_WAIT) -> dict:
    """Return {ok, running, started, url, error?} without raising.

    Never starts a second helper. If this process already spawned one, or the
    port is taken (models still loading, /health blocked), we wait instead of
    launching another python.exe.
    """
    url = helper_url(config)
    if health(url):
        return {"ok": True, "running": True, "started": False, "url": url}

    global _started
    we_spawned = False
    proc = None
    with _spawn_lock:
        if health(url):
            return {"ok": True, "running": True, "started": False, "url": url}
        owned_alive = _started is not None and _started.poll() is None
        if owned_alive:
            proc = _started
        elif _port_open(url):
            proc = None
        else:
            try:
                proc = spawn(config)
            except ValueError as exc:
                return {"ok": False, "running": False, "started": False, "url": url, "error": str(exc)}
            except OSError as exc:
                return {"ok": False, "running": False, "started": False, "url": url, "error": str(exc)}
            _started = proc
            we_spawned = True

    deadline = time.monotonic() + max(1.0, float(wait))
    while time.monotonic() < deadline:
        if health(url, timeout=0.8):
            return {"ok": True, "running": True, "started": we_spawned, "url": url}
        if proc is not None and proc.poll() is not None:
            if _started is proc:
                _started = None
            return {
                "ok": False,
                "running": False,
                "started": False,
                "url": url,
                "error": f"Avatar helper exited immediately (code {proc.returncode}). "
                         f"See lst-spawn.log in the helper folder.",
            }
        time.sleep(0.4)
    if health(url, timeout=0.8):
        return {"ok": True, "running": True, "started": we_spawned, "url": url}
    if _port_open(url) or (proc is not None and proc.poll() is None):
        return {"ok": True, "running": True, "started": we_spawned, "url": url}
    return {
        "ok": False,
        "running": False,
        "started": we_spawned,
        "url": url,
        "error": "Avatar helper was started but /health did not answer in time.",
    }


def stop_owned():
    """Stop the helper only if ``ensure_started`` spawned it in this process.

    Returns True if a live child was asked to exit. An already-running helper
    we did not start is never touched.
    """
    global _started
    proc = _started
    _started = None
    if proc is None:
        return False
    if proc.poll() is not None:
        return False
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        return False
    return True


atexit.register(stop_owned)


def read_voice_status(config) -> dict:
    """Read the helper's declared ready flag from disk, then /ready if needed."""
    folder = Path(str((config or {}).get("avatar_dir") or "").strip())
    path = folder / "voice-status.json" if str(folder) else None
    if path and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["ok"] = True
                data["ready"] = bool(data.get("ready"))
                data["missing"] = False
                if data["ready"]:
                    return data
        except Exception:
            pass
    try:
        code, raw = _request(helper_url(config) + "/ready", timeout=2.0)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        data["ok"] = 200 <= code < 300
        data["ready"] = bool(data.get("ready"))
        data["missing"] = False
        return data
    except Exception:
        return {
            "ok": False,
            "ready": False,
            "missing": True,
            "detail": "The helper has not published ready yet.",
        }


def _wait_until_down(url, wait=10.0):
    deadline = time.monotonic() + max(1.0, float(wait))
    while time.monotonic() < deadline:
        if not health(url, timeout=0.6):
            return True
        time.sleep(0.3)
    return not health(url, timeout=0.6)


def restart(config, wait: float = SPAWN_WAIT) -> dict:
    """Stop the current helper (owned child, or POST /shutdown) then start a new one."""
    url = helper_url(config)
    owned = _started is not None and _started.poll() is None
    if owned:
        stop_owned()
    elif health(url):
        try:
            _request(url + "/shutdown", method="POST", body={}, timeout=2)
        except Exception:
            pass
    if not _wait_until_down(url, wait=10):
        return {
            "ok": False,
            "running": True,
            "started": False,
            "url": url,
            "error": "The helper is still running. Close its window, then click Restart again.",
        }
    return ensure_started(config, wait=wait)


def rpc(config, path, method="GET", body=None, timeout=20):
    """Call the helper from this process. The browser must not talk to :8765 itself."""
    path = "/" + str(path or "").lstrip("/")
    method = str(method or "GET").upper()
    allowed = RPC_ALLOWED.get(path)
    if allowed != method:
        raise ValueError(f"Avatar path not allowed: {method} {path}")
    url = helper_url(config) + path
    try:
        code, raw = _request(url, method=method, body=body, timeout=timeout)
        data = json.loads(raw)
        if 200 <= code < 300:
            return data
        raise RuntimeError(data.get("error") or raw or f"HTTP {code}")
    except HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8")
            parsed = json.loads(raw)
            err = parsed.get("error") or raw or str(exc)
        except Exception:
            err = raw or str(exc)
        raise RuntimeError(err) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Avatar helper at {url} did not return JSON. Is the Helper URL in Settings "
            f"pointing at the Avatar Read Server (port 8765), not this app?"
        ) from exc
    except (URLError, OSError, TimeoutError) as exc:
        why = getattr(exc, "reason", None) or exc
        raise RuntimeError(
            f"Cannot reach the Avatar helper at {url} ({why}). "
            "Open Settings → Voice / Avatar helper, confirm the folder, then "
            "click Open Avatar settings. If that tab fails, the helper is not running."
        ) from exc
