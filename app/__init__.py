"""Local Streaming Token — by Assisted Intel.

A local Ollama chat client with a web-browser GUI. This package contains the
backend (Ollama client, web search), the GUI-free prompt-assembly logic, the
JSON persistence store, the native OS file/folder dialogs, and the Flask server.
"""

from .core import APP_NAME, APP_AUTHOR, APP_VERSION

__all__ = ["APP_NAME", "APP_AUTHOR", "APP_VERSION"]
