#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Backend core for the Ollama chat client. Contains everything that is GUI-free:
- OllamaClient (list models, capabilities, streaming chat with tool-calling)
- Web search pipeline (Brave discovery + Bright Data crawling)
- Text helpers (strip_markdown, strip_reasoning, reasoning-model heuristics)
- JSON persistence helpers (load_json / save_json)
- Library (Resources) helpers + XML import/export

This module was extracted verbatim from the original wxPython app so behaviour
is identical; only the wx colour palette was removed.
"""

import requests
import json
import os
import re
import sys
import threading
import uuid
import html as html_lib
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from . import crypto


# --------------------------- Broken optional dependencies ---------------------------

class _BrokenModuleBlocker:
    """A ``sys.meta_path`` finder that fails a known-broken module instantly.

    Sits at the front of the meta path and raises ``ModuleNotFoundError`` for the named
    packages (and their submodules) before any filesystem search happens.
    """

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names:
            raise ModuleNotFoundError(
                f"{root} is installed but unimportable and has been disabled by "
                f"{APP_NAME}; reinstall it to use features that need it.",
                name=fullname)
        return None     # everything else: fall through to the normal finders


def _neutralize_broken_optional_imports():
    """Stop libraries paying for an optional dependency that is installed but unimportable.

    DuckDB (and LanceDB, pyarrow, and friends) probe for pandas/numpy/pyarrow while
    converting values. When such a package is INSTALLED BUT BROKEN — classically a
    numpy/pandas ABI mismatch ("numpy.dtype size changed"), or a bad DLL directory —
    the import raises something other than ImportError, and CPython does not cache
    failed imports. So every probe rescans sys.path and re-executes part of the package.

    Measured here on a conda base env with a mismatched pandas: DuckDB attempted 5,280
    pandas imports for a 40-row insert, and a 720-row insert took 301 SECONDS. Blocking
    the import took the same workload from ~3 rows/s to ~800 rows/s.

    We block via a meta-path finder rather than the more obvious
    ``sys.modules[name] = None``. That idiom does make ``import name`` fail, but it also
    makes the widely-used polars-style lazy-import shim believe the module is *loaded*::

        if module_name in sys.modules:
            return sys.modules[module_name], True    # -> (None, "available")

    LanceDB uses exactly that shim, and would then dereference ``None.__version__``.
    Raising ModuleNotFoundError from ``find_spec`` instead leaves ``sys.modules`` clean,
    so such probes correctly conclude the module is unavailable — and it is still
    instant, because no path search runs.

    A genuinely absent package is left alone: CPython already caches that cheaply.
    Returns ``[(name, error)]`` so startup can warn about what it disabled.
    """
    broken = []
    for mod in ("pandas", "numpy", "pyarrow"):
        if mod in sys.modules:
            continue
        try:
            __import__(mod)
        except ImportError:
            pass            # not installed — nothing to do, and the failure is cached
        except Exception as e:
            broken.append((mod, f"{type(e).__name__}: {e}"))
    if broken:
        # Drop any partially-initialised submodules the failed import left behind,
        # or the blocker will be bypassed for those names.
        roots = {name for name, _ in broken}
        for key in [k for k in sys.modules
                    if k.split(".")[0] in roots]:
            del sys.modules[key]
        sys.meta_path.insert(0, _BrokenModuleBlocker(roots))
    return broken


BROKEN_OPTIONAL_IMPORTS = _neutralize_broken_optional_imports()

# --------------------------- Branding ---------------------------
APP_NAME = "Local Streaming Token"
APP_AUTHOR = "Assisted Intel"
APP_VERSION = "2.0.0"

# --------------------------- Provider presets ---------------------------
# Each preset maps a user-facing provider choice to an adapter `type`
# (ollama | anthropic | openai) and a default base_url. The "openai" type is the
# OpenAI-compatible adapter and covers OpenAI, xAI, Gemini, DeepSeek, and friends.
PROVIDER_PRESETS = [
    {"key": "ollama",     "label": "Ollama (local network)", "type": "ollama",    "base_url": "http://127.0.0.1:11434", "needs_key": False},
    {"key": "anthropic",  "label": "Anthropic (Claude)",    "type": "anthropic", "base_url": "https://api.anthropic.com",                 "needs_key": True},
    {"key": "openai",     "label": "OpenAI (GPT)",          "type": "openai",    "base_url": "https://api.openai.com/v1",                 "needs_key": True},
    {"key": "xai",        "label": "xAI (Grok)",            "type": "openai",    "base_url": "https://api.x.ai/v1",                       "needs_key": True},
    {"key": "gemini",     "label": "Google Gemini",         "type": "openai",    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "needs_key": True},
    {"key": "deepseek",   "label": "DeepSeek",              "type": "openai",    "base_url": "https://api.deepseek.com/v1",               "needs_key": True},
    {"key": "groq",       "label": "Groq",                  "type": "openai",    "base_url": "https://api.groq.com/openai/v1",            "needs_key": True},
    {"key": "mistral",    "label": "Mistral",               "type": "openai",    "base_url": "https://api.mistral.ai/v1",                 "needs_key": True},
    {"key": "openrouter", "label": "OpenRouter",            "type": "openai",    "base_url": "https://openrouter.ai/api/v1",              "needs_key": True},
    {"key": "lmstudio",   "label": "LM Studio (local)",     "type": "openai",    "base_url": "http://localhost:1234/v1",                  "needs_key": False},
    {"key": "custom",     "label": "Custom OpenAI-compatible", "type": "openai", "base_url": "",                                         "needs_key": True},
]

# --------------------------- Config / Data ---------------------------

DEFAULT_LOCAL_URL = "http://127.0.0.1:11434"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# User data (chats/presets/libraries) lives in data/; all configuration and
# secrets (API keys, servers/providers) live in the gitignored settings/ folder.
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SETTINGS_DIR = PROJECT_ROOT / "settings"
SETTINGS_DIR.mkdir(exist_ok=True)
LEGACY_CONFIG_FILE = DATA_DIR / "config.json"   # pre-settings/ location (migrated on first run)

# App-wide encryption keyfile (NOT per-profile): holds the password-wrapped Data
# Encryption Key + the Flask session secret. See app/crypto.py.
APP_KEYFILE = SETTINGS_DIR / "app_key.enc"

# --- Profiles ---------------------------------------------------------------
# The app supports two independent axes of profiles, each a folder of files:
#   * DATA profiles     -> data/profiles/<id>/     (chats, prompts, resources,
#                          evals, database sessions + credential vault, RAG store)
#   * SETTINGS profiles -> settings/profiles/<id>/ (settings.json: providers, keys,
#                          tokens, defaults)
# The per-profile file paths below are MODULE GLOBALS reassigned by the two setters
# whenever a profile is activated. Downstream code always dereferences core.<CONST>
# at call time, so reassigning the globals redirects reads/writes with no other
# changes. They are seeded to the legacy flat locations so imports have valid values
# before create_app() activates the real profile at startup.
DATA_PROFILES_DIR = DATA_DIR / "profiles"
SETTINGS_PROFILES_DIR = SETTINGS_DIR / "profiles"
DATA_REGISTRY_FILE = DATA_DIR / "profiles.json"          # {active, profiles:[{id,name}]}
SETTINGS_REGISTRY_FILE = SETTINGS_DIR / "profiles.json"
INCOGNITO_DIR = DATA_PROFILES_DIR / ".incognito"         # scratch dir for the private session

# Basenames of the per-profile data files (relative to a data-profile folder).
DATA_FILE_NAMES = ("chats.json", "chat_groups.json", "presets.json", "prompts.json",
                   "libraries.json", "evals.json", "db_projects.json", "context_history.json")

SETTINGS_FILE = SETTINGS_DIR / "settings.json"
CHATS_FILE = DATA_DIR / "chats.json"
CHAT_GROUPS_FILE = DATA_DIR / "chat_groups.json"  # sidebar tabs for imported chats: [{id, name}]
PRESETS_FILE = DATA_DIR / "presets.json"
PROMPTS_FILE = DATA_DIR / "prompts.json"       # System/Pre-prompt library: Group -> Category -> Prompt trees
LIBRARIES_FILE = DATA_DIR / "libraries.json"
EVALS_FILE = DATA_DIR / "evals.json"           # Prompt Validation & Evaluation projects
CONTEXT_HISTORY_FILE = DATA_DIR / "context_history.json"  # per-chat LLM-call token-usage history {chat_id: [entry, ...]}
RAG_DB_FILE = DATA_DIR / "rag.duckdb"          # persistent RAG vector store (library chunk embeddings)
# LanceDB alternative to RAG_DB_FILE — a *directory*, and PLAINTEXT (chunk text and
# embeddings are not encrypted). Both stores can exist side by side; Settings → RAG
# chooses which one is live. Named *.lance so the .gitignore rule catches it wherever
# it ends up.
RAG_LANCE_DIR = DATA_DIR / "rag.lance"
COMPILED_FILE = DATA_DIR / "compiled.json"     # "Compile Data" manifests (rebuildable cache): {"library:<id>"/"persona:<id>": {...}}
PERSONAS_DIR = DATA_DIR / "personas"           # one <persona_id>/ folder each (persona.xml + sources/ + memories/)

# --- Database Processing tab ------------------------------------------------
# Non-secret session/project metadata lives with the data profile (like the other
# *.json). Connection credentials NEVER live here — they go in the AES-GCM vault.
DB_PROJECTS_FILE = DATA_DIR / "db_projects.json"   # import sessions + column configs (no secrets)
DB_DIR = DATA_DIR / "db"                            # DuckDB staging files + audit logs
DB_STAGING_DIR = DB_DIR / "staging"                 # one staging_<session>.duckdb per import
DB_AUDIT_DIR = DB_DIR / "audit"                     # one <session>.jsonl append-only log per session
VAULT_FILE = DATA_DIR / "db_vault.enc"              # AES-GCM blob of connection profiles (per data profile)


def set_active_data_profile(profile_dir):
    """Point every per-data-profile path at ``profile_dir`` and ensure its subdirs
    exist. Reassigns module globals in place (callers read core.<CONST> at call time)."""
    global CHATS_FILE, CHAT_GROUPS_FILE, PRESETS_FILE, PROMPTS_FILE, LIBRARIES_FILE
    global EVALS_FILE, CONTEXT_HISTORY_FILE, DB_PROJECTS_FILE, RAG_DB_FILE, RAG_LANCE_DIR
    global COMPILED_FILE, PERSONAS_DIR
    global DB_DIR, DB_STAGING_DIR, DB_AUDIT_DIR, VAULT_FILE
    d = Path(profile_dir)
    d.mkdir(parents=True, exist_ok=True)
    CHATS_FILE = d / "chats.json"
    CHAT_GROUPS_FILE = d / "chat_groups.json"
    PRESETS_FILE = d / "presets.json"
    PROMPTS_FILE = d / "prompts.json"
    LIBRARIES_FILE = d / "libraries.json"
    EVALS_FILE = d / "evals.json"
    CONTEXT_HISTORY_FILE = d / "context_history.json"
    DB_PROJECTS_FILE = d / "db_projects.json"
    RAG_DB_FILE = d / "rag.duckdb"
    RAG_LANCE_DIR = d / "rag.lance"
    COMPILED_FILE = d / "compiled.json"
    PERSONAS_DIR = d / "personas"
    DB_DIR = d / "db"
    DB_STAGING_DIR = DB_DIR / "staging"
    DB_AUDIT_DIR = DB_DIR / "audit"
    VAULT_FILE = d / "db_vault.enc"
    for _d in (DB_DIR, DB_STAGING_DIR, DB_AUDIT_DIR):
        _d.mkdir(parents=True, exist_ok=True)


def set_active_settings_profile(profile_dir):
    """Point SETTINGS_FILE at ``profile_dir/settings.json``."""
    global SETTINGS_FILE
    d = Path(profile_dir)
    d.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE = d / "settings.json"

# Standard context lengths for the dropdown (up to 256K)
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# --------------------------- Web Search ---------------------------
# Brave Search API is used to discover result URLs + rich metadata (descriptions,
# snippets). Bright Data's Web Unlocker then crawls the full page text.
# Credentials are NOT stored in source (this repo is publishable) — they live in
# the gitignored settings/settings.json and are injected via configure_tokens().
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_TOKEN = ""

BRIGHTDATA_ENDPOINT = "https://api.brightdata.com/request"
BRIGHTDATA_TOKEN = ""
BRIGHTDATA_ZONE = "web_unlocker1"


def configure_tokens(brave_token=None, brightdata_token=None, brightdata_zone=None):
    """Override the web-search credentials at runtime (from config.json)."""
    global BRAVE_TOKEN, BRIGHTDATA_TOKEN, BRIGHTDATA_ZONE
    if brave_token:
        BRAVE_TOKEN = brave_token
    if brightdata_token:
        BRIGHTDATA_TOKEN = brightdata_token
    if brightdata_zone:
        BRIGHTDATA_ZONE = brightdata_zone


# Web crawl tuning: how many fully-crawled pages to gather before answering.
MIN_CRAWLED_PAGES = 7        # require at least this many completed page crawls
MAX_CRAWL_CANDIDATES = 30    # most search results we will attempt to crawl
CRAWL_WORKERS = 6            # parallel page fetches
PER_PAGE_CHARS = 2200        # cap on extracted text kept per page
MIN_PAGE_CHARS = 400         # a page shorter than this is treated as not-crawled

# Markers that indicate a login wall / paywall / bot-block rather than real content.
BLOCKED_PAGE_MARKERS = (
    "log in to view", "login to view", "log in to continue", "sign in to continue",
    "please log in", "please sign in", "sign in to read", "log in to read",
    "you must be logged in", "subscribe to read", "subscribe to continue",
    "create an account to continue", "create a free account", "register to continue",
    "enable javascript", "please enable javascript", "javascript is required",
    "access denied", "are you a robot", "verify you are human", "unusual traffic",
    "this content is available to subscribers", "become a member to",
    "captcha", "cf-browser-verification", "just a moment...",
)

# Tool definition advertised to tool-capable Ollama models.
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web and crawl the full text of the top result pages "
            "for current, factual, or up-to-date information. Use this whenever the "
            "user asks about recent events, live data, or anything that may have "
            "changed after training."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased as you would type it into a search engine.",
                },
            },
            "required": ["query"],
        },
    },
}


# Bright Data returns a 200 with one of these bodies when the account/zone itself
# fails (not the target site). Detect these so we can report a clear reason.
BRIGHTDATA_ACCOUNT_ERRORS = (
    "zone has reached usage limit",
    "the webpage could not be loaded because",
    "no available ips",
    "zone_gen_failed",
    "invalid auth",
)


def _brightdata_fetch(url: str, timeout: int = 60) -> str:
    """Fetch a URL through Bright Data's Web Unlocker and return the raw body.
    Raises RuntimeError('Bright Data: ...') when the account/zone itself errors.
    """
    resp = requests.post(
        BRIGHTDATA_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BRIGHTDATA_TOKEN}",
        },
        json={"zone": BRIGHTDATA_ZONE, "url": url, "format": "raw"},
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.text
    low = text[:400].lower()
    if any(marker in low for marker in BRIGHTDATA_ACCOUNT_ERRORS):
        reason = _html_to_text(text)[:200] or "unknown error"
        raise RuntimeError(f"Bright Data: {reason}")
    return text


def _html_to_text(fragment: str) -> str:
    """Strip HTML tags + unescape entities + collapse whitespace. Strips tags a
    second time after unescaping to catch entity-encoded tags like &lt;br /&gt;.
    """
    if not fragment:
        return ""
    text = re.sub(r"<[^>]+>", "", fragment)
    text = html_lib.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _brave_search_results(query: str, limit: int = MAX_CRAWL_CANDIDATES):
    """Query the Brave Search API and return up to `limit` result dicts, each with
    url / title / description / snippets. Paginates via `offset` as needed.
    """
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_TOKEN,
    }
    seen = set()
    out = []
    offset = 0
    while len(out) < limit and offset <= 9:  # Brave allows offset 0..9
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers=headers,
            params={
                "q": query,
                "count": 20,           # max results per page
                "offset": offset,
                "result_filter": "web",
                "extra_snippets": 1,   # include extra text snippets (paid plans)
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = (resp.json().get("web") or {}).get("results") or []
        if not results:
            break
        for r in results:
            url = r.get("url")
            if not url or not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            out.append({
                "url": url,
                "title": _html_to_text(r.get("title") or "") or url,
                "description": _html_to_text(r.get("description") or ""),
                "snippets": [_html_to_text(s) for s in (r.get("extra_snippets") or []) if s],
            })
            if len(out) >= limit:
                break
        offset += 1
    return out


# Element/attribute smells that mark page chrome (menus, rails, footers, share bars)
# rather than article content. Word-boundaried so e.g. "ad" never matches "read"/"header".
_CHROME_ROLES = {"navigation", "banner", "complementary", "contentinfo", "search", "menu",
                 "menubar", "tablist", "dialog"}
_CHROME_ATTR_RE = re.compile(
    r"(?i)(?:^|[-_ ])(?:nav|navbar|menu|sidebar|side-?bar|footer|header|masthead|breadcrumb|"
    r"share|social|comment|related|recommend|promo|advert|\bads?\b|cookie|consent|banner|"
    r"newsletter|subscribe|signup|toolbar|pagination|paginate|widget|byline|skip-link|"
    r"site-?nav|global-?nav|topbar|utility|disclaimer)"
)

# Block-level tags that hold real prose — extracted one line each so sentences stay whole.
_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "dd", "dt", "pre"]

# Section headings whose content is citation/navigation noise, not article prose.
_STOP_HEADS = {"references", "notes", "external links", "further reading", "see also",
               "bibliography", "sources", "citations", "footnotes", "works cited",
               "navigation menu"}

# CSS selectors for citation / navigation / boilerplate containers (mostly MediaWiki, but
# harmless on other sites). Removed before extraction so reference lists, "See also", nav
# boxes, edit links, and cite templates never reach the output.
_PRECLEAN_SELECTORS = [
    ".reflist", ".refbegin", ".references", "ol.references", ".mw-references-wrap",
    ".reference", "cite.citation", ".citation", "sup.reference", ".navbox",
    ".vertical-navbox", ".sistersitebox", ".catlinks", ".mw-editsection", ".noprint",
    ".mw-jump-link", ".mw-indicators", ".shortdescription", ".hatnote", ".ambox",
    ".metadata", ".portal",
]


def _norm_lines(text: str) -> str:
    """Normalize extracted text to clean, blank-line-separated paragraphs: strip zero-width /
    RTL marks, collapse whitespace, fix spaces before punctuation, drop tiny/URL-only lines."""
    if not text:
        return ""
    text = re.sub(r"[​‎‏﻿­]", "", text)   # zero-width / RTL / soft-hyphen
    paras = []
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        ln = re.sub(r"\s+([,.;:!?%])", r"\1", ln)                  # space before punctuation
        if len(ln) < 3 or re.match(r"^https?://\S+$", ln):        # tiny fragment or bare URL
            continue
        paras.append(ln)
    return "\n\n".join(paras).strip()


def _strip_stop_sections(soup) -> None:
    """Remove trailing citation/navigation sections (References, See also, …). Handles Parsoid
    HTML (heading wrapped in <section> → drop the section) and classic MediaWiki/other HTML
    (drop the heading + following siblings up to the next same-or-higher heading)."""
    for h in soup.find_all(["h2", "h3"]):
        if h.decomposed:
            continue
        title = re.sub(r"\[edit\]", "", h.get_text(" ", strip=True)).strip().lower()
        if title not in _STOP_HEADS:
            continue
        sec = h.find_parent("section")
        if sec is not None:
            sec.decompose()
            continue
        node = h
        if h.parent is not None and "mw-heading" in (h.parent.get("class") or []):
            node = h.parent
        for sib in list(node.find_next_siblings()):
            if sib.name in ("h1", "h2") or (sib.name == "h3" and h.name == "h3"):
                break
            sib.decompose()
        node.decompose()


def _preclean_soup(soup):
    """Strip citation/nav/boilerplate containers + stop-sections from a parsed soup (mutates)."""
    for sel in _PRECLEAN_SELECTORS:
        try:
            for el in soup.select(sel):
                if not el.decomposed:
                    el.decompose()
        except Exception:
            pass
    _strip_stop_sections(soup)
    return soup


def _preclean_html(html: str) -> str:
    """Parse HTML, strip citation/nav boilerplate, and re-serialize (input for trafilatura)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    _preclean_soup(soup)
    return str(soup)


def _extract_page_text_regex(html: str) -> str:
    """Regex-only extractor (last-resort fallback when bs4/trafilatura are unavailable).
    Drops scripts/styles/nav/header/footer/aside chrome and prefers the main article region.
    """
    if not html:
        return ""
    html = re.sub(r"(?is)<script.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?</noscript>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    # Drop obvious chrome so menus/footers don't crowd out the real content.
    html = re.sub(r"(?is)<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>", " ", html)

    # Prefer the main article region if the page marks one up.
    m = re.search(r"(?is)<(article|main)\b[^>]*>(.*?)</\1>", html)
    if m and len(m.group(2)) > 500:
        html = m.group(2)

    # Turn block-level closers into newlines so paragraphs survive.
    html = re.sub(r"(?i)</?(p|div|section|article|h[1-6]|li|tr|br)[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = html_lib.unescape(text)
    return _norm_lines(text)


def _extract_page_text_bs4(html: str) -> str:
    """BeautifulSoup block extractor: drop chrome + citation/nav boilerplate, then emit one
    line per block element so inline links collapse into whole sentences. Used as the fallback
    when trafilatura isn't available."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements (incl. images/tables/figures → drops captions + data dumps).
    for tag in soup(["script", "style", "noscript", "template", "svg", "form", "button",
                     "input", "select", "textarea", "iframe", "nav", "header", "footer",
                     "aside", "figure", "figcaption", "img", "table", "audio", "video",
                     "dialog"]):
        tag.decompose()
    _preclean_soup(soup)

    # Remove chrome by ARIA role / class-id smell, protecting the main content wrapper.
    main_marker = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
    protected = set()
    if main_marker is not None:
        protected = {id(main_marker)} | {id(a) for a in main_marker.parents}
    for el in soup.find_all(True):
        if el.decomposed or id(el) in protected:
            continue
        # Never drop structural wrappers via a class smell — a chrome-ish class on <body>
        # (e.g. WordPress "single-post …-header …") would otherwise nuke the whole page.
        if el.name in ("html", "body", "main", "article"):
            continue
        role = (el.get("role") or "").strip().lower()
        attr_blob = " ".join(el.get("class") or []) + " " + (el.get("id") or "")
        if role in _CHROME_ROLES or (attr_blob.strip() and _CHROME_ATTR_RE.search(attr_blob)):
            el.decompose()

    main = soup.body or soup
    parts = []
    for el in main.find_all(_BLOCK_TAGS):
        if el.decomposed:
            continue
        # Skip a block nested inside another captured block (avoids duplicate text); keep <li>.
        if el.name != "li" and el.find_parent(_BLOCK_TAGS) is not None:
            continue
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = _norm_lines("\n\n".join(parts))
    # Some sites wrap paragraphs in <div>s (not <p>); block extraction then comes up thin.
    # Fall back to the full cleaned text so that content isn't lost.
    if len(text) < MIN_PAGE_CHARS:
        text = _norm_lines(main.get_text("\n"))
    return text


def _extract_page_text(html: str) -> str:
    """Extract just the readable article text from a raw HTML page, as clean paragraphs.

    Tier 1 uses ``trafilatura`` (robust main-content detection — drops nav menus, headers,
    footers, sidebars, related rails, infoboxes, and tables) on HTML that has first been
    pre-cleaned of citation/reference boilerplate. Tier 2 is a BeautifulSoup block extractor,
    tier 3 a regex extractor — each used only if the previous is unavailable or comes back too
    thin, so the crawl success rate never regresses.
    """
    if not html:
        return ""

    candidates = []

    # Tier 1: trafilatura (best article isolation + paragraph output).
    try:
        import trafilatura
        out = trafilatura.extract(
            _preclean_html(html), include_comments=False, include_tables=False,
            include_links=False, include_formatting=False, include_images=False,
            favor_precision=True, output_format="txt") or ""
        out = _norm_lines(out)
        if len(out) >= MIN_PAGE_CHARS:
            return out
        if out:
            candidates.append(out)
    except Exception:
        pass

    # Tier 2: BeautifulSoup block extractor.
    try:
        out = _extract_page_text_bs4(html)
        if len(out) >= MIN_PAGE_CHARS:
            return out
        if out:
            candidates.append(out)
    except Exception:
        pass

    # Tier 3: regex last resort. Return the longest thing we got (safeguard for thin pages).
    candidates.append(_extract_page_text_regex(html))
    return max(candidates, key=len) if candidates else ""


def _is_blocked_or_incomplete(text: str) -> bool:
    """True if the crawl didn't yield real content (login wall, paywall, bot-block,
    JS-only page, or simply too little text). Such pages do not count toward the goal.
    """
    if not text or len(text) < MIN_PAGE_CHARS:
        return True
    head = text[:6000].lower()
    return any(marker in head for marker in BLOCKED_PAGE_MARKERS)


def _normalize_domain(text: str) -> str:
    """Reduce user input to a bare hostname (drops scheme, path, and leading www.)."""
    d = (text or "").strip().lower()
    d = re.sub(r"^\w+://", "", d)   # strip scheme
    d = d.split("/")[0].split("?")[0].strip().strip(".")
    if d.startswith("www."):
        d = d[4:]
    return d


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except Exception:
        return ""


def _host_in_domains(host: str, domains) -> bool:
    """True if host equals or is a subdomain of any approved domain."""
    host = (host or "").lower()
    for d in domains:
        d = d.lower()
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def web_search(query: str, min_pages: int = MIN_CRAWLED_PAGES, should_stop=None,
               allowed_domains=None) -> str:
    """Search via Brave, then crawl full pages via Bright Data until at least
    `min_pages` pages have been *completely* crawled. Pages behind a login/paywall,
    blocked by bot protection, or that return too little text are skipped and do
    not count toward the goal.

    Returns the crawled page contents (title, URL, Brave summary, body text), plus
    a short list of the remaining Brave results as snippet-only extra context.
    """
    query = (query or "").strip()
    if not query:
        return "No search query was provided."

    def stopped():
        return bool(should_stop and should_stop())

    # Optional restriction to an approved list of domains.
    domains = [d for d in (allowed_domains or []) if d]
    brave_query = query
    if domains:
        # Brave supports the site: operator; OR them together to allow any domain.
        brave_query = f"{query} (" + " OR ".join(f"site:{d}" for d in domains) + ")"

    # Gather more candidates than needed since some pages will be blocked/empty.
    candidate_limit = max(MAX_CRAWL_CANDIDATES, min_pages * 4)
    try:
        candidates = _brave_search_results(brave_query, limit=candidate_limit)
    except Exception as e:
        return f"Web search failed (Brave Search API error): {e}"

    # Safety net: even if Brave returns off-domain results, keep only approved ones.
    if domains:
        candidates = [c for c in candidates if _host_in_domains(_url_host(c["url"]), domains)]

    if not candidates:
        if domains:
            return (f"No results within the approved domains ({', '.join(domains)}) "
                    f"were found for '{query}'.")
        return f"No search results were found for '{query}'."

    crawler_error = {"msg": None}

    def crawl(item):
        try:
            html = _brightdata_fetch(item["url"], timeout=45)
        except Exception as e:
            if str(e).startswith("Bright Data:"):
                crawler_error["msg"] = str(e)
            return None
        text = _extract_page_text(html)
        if _is_blocked_or_incomplete(text):
            return None
        return {**item, "text": text[:PER_PAGE_CHARS]}

    completed = []
    crawled_urls = set()
    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as ex:
        futures = {ex.submit(crawl, item): item for item in candidates}
        for fut in as_completed(futures):
            if stopped():
                break
            page = fut.result()
            if page:
                completed.append(page)
                crawled_urls.add(page["url"])
            if len(completed) >= min_pages:
                break

    if not completed:
        reason = (
            f"the page crawler is unavailable ({crawler_error['msg']})"
            if crawler_error["msg"]
            else "the pages were login-walled, bot-blocked, or empty"
        )
        return (
            f"Searched the web for '{query}' via Brave, but no pages could be fully "
            f"crawled because {reason}. Brave search results (snippets only):\n\n"
            + "\n".join(
                f"- {c['title']} — {c['url']}\n  {c['description'] or (c['snippets'][0] if c['snippets'] else '')}"
                for c in candidates[:max(min_pages, 5)]
            )
        )

    restrict_note = f" Restricted to approved domains: {', '.join(domains)}." if domains else ""
    header = (
        f"Web research for query: \"{query}\".{restrict_note}\n"
        f"{len(completed)} fully-crawled page(s) "
        f"(target was {min_pages}; login-walled/blocked pages were skipped):\n"
    )
    blocks = []
    for i, p in enumerate(completed, 1):
        summary = f"SUMMARY: {p['description']}\n\n" if p.get("description") else ""
        blocks.append(
            f"===== PAGE {i} =====\nTITLE: {p['title']}\nURL: {p['url']}\n\n{summary}{p['text']}"
        )

    note = ""
    if len(completed) < min_pages:
        note = (
            f"\n\n(NOTE: only {len(completed)} of the target {min_pages} pages could be "
            "fully crawled; the rest were login-walled, blocked, or empty.)"
        )

    # Add a few remaining Brave results as snippet-only extra context.
    extras = [c for c in candidates if c["url"] not in crawled_urls][:5]
    extra_block = ""
    if extras:
        lines = []
        for c in extras:
            snip = c["description"] or (c["snippets"][0] if c["snippets"] else "")
            lines.append(f"- {c['title']} — {c['url']}\n  {snip}")
        extra_block = "\n\nADDITIONAL SEARCH RESULTS (not fully crawled, snippets only):\n" + "\n".join(lines)

    return header + "\n\n" + "\n\n".join(blocks) + note + extra_block


# --------------------------- Single-URL / library crawling ---------------------------
# Used by the Resources (Libraries) tab to scrape a page into an editable item. The
# fetch tries Bright Data first (best at bypassing bot-blocks), then a local headless
# Playwright browser (renders JS), then a plain requests GET as a last resort — so it
# still works before `playwright install` has been run, or with no Bright Data token.

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Library scrapes keep far more text than a chat web-search snippet: the content is
# chunked + embedded by "Compile Data", so the fuller the article the better.
LIBRARY_PAGE_CHARS = 200_000


def _title_from_html(html: str, fallback: str = "") -> str:
    """Best-effort page title from a raw HTML string, else the fallback."""
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    if m:
        t = _html_to_text(m.group(1))
        if t:
            return t[:200]
    return fallback


def _playwright_fetch(url: str, timeout: int = 45) -> str:
    """Render a page in a local headless Chromium and return its HTML.

    Raises RuntimeError with an actionable message when Playwright (or its browser
    binary) isn't installed, so the caller can fall through to a plain requests GET.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        raise RuntimeError(
            "Playwright is not installed. Install it with:  pip install playwright  "
            "then:  playwright install chromium")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=_DEFAULT_UA)
                page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                return page.content()
            finally:
                browser.close()
    except Exception as e:
        msg = str(e)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            raise RuntimeError(
                "Playwright's Chromium browser isn't installed. Run:  "
                "playwright install chromium")
        raise RuntimeError(f"Playwright crawl failed: {msg}")


def _requests_fetch(url: str, timeout: int = 45) -> str:
    """Plain HTTP GET returning the response body (last-resort fallback)."""
    resp = requests.get(url, headers={"User-Agent": _DEFAULT_UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_url_text(url: str, timeout: int = 45) -> dict:
    """Scrape a single URL to readable text using the Bright Data → Playwright →
    requests fallback chain. Returns {url, title, text, via}. Raises RuntimeError
    only when every tier fails or none yield real content.
    """
    url = (url or "").strip()
    if not url:
        raise RuntimeError("No URL was provided.")
    if not re.match(r"(?i)^https?://", url):
        url = "https://" + url

    attempts = []  # (name, callable)
    if BRIGHTDATA_TOKEN:
        attempts.append(("Bright Data", lambda: _brightdata_fetch(url, timeout=timeout)))
    attempts.append(("Playwright", lambda: _playwright_fetch(url, timeout=timeout)))
    attempts.append(("requests", lambda: _requests_fetch(url, timeout=timeout)))

    errors = []
    for name, fn in attempts:
        try:
            html = fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        text = _extract_page_text(html)
        if _is_blocked_or_incomplete(text):
            errors.append(f"{name}: page was blocked, empty, or too short")
            continue
        return {
            "url": url,
            "title": _title_from_html(html, fallback=_url_host(url) or url),
            "text": text[:LIBRARY_PAGE_CHARS],
            "via": name,
        }
    raise RuntimeError(
        f"Could not fetch '{url}'. Tried: " + " | ".join(errors))


def crawl_search(query, sites=None, max_results=5, should_stop=None):
    """Discover URLs via Brave, then crawl each with the fetch fallback chain until
    `max_results` pages crawl successfully (or candidates run out).

    A generator that yields progress dicts as it works and a final result dict:
        {"type": "progress", "done": int, "target": int, "url": str, "title": str, "ok": bool}
        {"type": "result",   "pages": [{"title", "url", "text"}, ...],
                             "attempted": int, "errors": [str, ...]}

    Streamed so the UI can show live progress. Reuses the same Brave discovery,
    domain filtering, and text extraction as web_search().
    """
    def stopped():
        return bool(should_stop and should_stop())

    query = (query or "").strip()
    if not query:
        yield {"type": "result", "pages": [], "attempted": 0,
               "errors": ["No search query was provided."]}
        return

    domains = [_normalize_domain(d) for d in (sites or []) if _normalize_domain(d)]
    brave_query = query
    if domains:
        brave_query = f"{query} (" + " OR ".join(f"site:{d}" for d in domains) + ")"

    try:
        max_results = max(1, int(max_results))
    except Exception:
        max_results = 5

    # Gather more candidates than needed since some pages will be blocked/empty.
    candidate_limit = max(MAX_CRAWL_CANDIDATES, max_results * 4)
    try:
        candidates = _brave_search_results(brave_query, limit=candidate_limit)
    except Exception as e:
        yield {"type": "result", "pages": [], "attempted": 0,
               "errors": [f"Brave Search API error: {e}"]}
        return

    if domains:
        candidates = [c for c in candidates
                      if _host_in_domains(_url_host(c["url"]), domains)]

    pages = []
    errors = []
    attempted = 0
    for c in candidates:
        if stopped() or len(pages) >= max_results:
            break
        attempted += 1
        try:
            page = fetch_url_text(c["url"])
        except Exception as e:
            errors.append(f"{c['url']}: {e}")
            yield {"type": "progress", "done": len(pages), "target": max_results,
                   "url": c["url"], "title": c.get("title", ""), "ok": False}
            continue
        # Prefer the Brave result title when the page didn't expose a usable one.
        page["title"] = page.get("title") or c.get("title") or page["url"]
        pages.append(page)
        yield {"type": "progress", "done": len(pages), "target": max_results,
               "url": page["url"], "title": page["title"], "ok": True}

    yield {"type": "result", "pages": pages, "attempted": attempted, "errors": errors}


def strip_markdown(text: str) -> str:
    """Strip common Markdown formatting so AI replies are plain text only."""
    if not text:
        return text

    # Remove fenced code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove headers (#, ##, etc.)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove bold (**text** and __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # Remove italic (*text* and _text_) - be careful not to break lists
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)

    # Remove list markers (-, *, +, numbered)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def strip_reasoning(text: str) -> str:
    """Strip a reasoning model's chain-of-thought so only the final answer remains.

    Handles models that inline their reasoning in the content as
    <think>...</think> (DeepSeek-R1 etc.), <thinking>...</thinking>, or
    <reasoning>...</reasoning>. Ollama's native `think` mode already returns
    reasoning in a separate field, so this is a fallback for models that emit
    the tags inline anyway (including responses cut off mid-thought).
    """
    if not text:
        return text

    # Remove complete reasoning blocks
    text = re.sub(r'(?is)<(think|thinking|reasoning)>.*?</\1>', '', text)

    # Remove a dangling opening tag with no close (e.g. stopped mid-thought)
    text = re.sub(r'(?is)<(think|thinking|reasoning)>.*$', '', text)

    return text.strip()


# Substrings that flag a reasoning/thinking model by name. Used only as a
# fallback when the Ollama server is too old to report model capabilities.
REASONING_NAME_HINTS = (
    "deepseek-r1", "-r1:", "/r1", "qwq", "reasoning", "-thinking", "thinker",
    "magistral", "phi4-reasoning", "phi-4-reasoning", "exaone-deep", "cogito",
    "marco-o1", "openthinker", "deepscaler",
)


def looks_like_reasoning_model(name: str) -> bool:
    """Heuristic guess from a model name (fallback when capabilities are absent)."""
    if not name:
        return False
    n = name.lower()
    return any(h in n for h in REASONING_NAME_HINTS)


# --------------------------- Encrypted file I/O ---------------------------
# Every data/preference file the app persists goes through these helpers, which
# transparently encrypt at rest with the active Data Encryption Key (see crypto.py).
# When the app is locked (before login) or the key is unavailable, files are written
# as plaintext and read as-is — the login-gated server never serves data while locked,
# and the first-run migration re-encrypts any legacy plaintext.

def read_bytes(path):
    """Read a file, transparently decrypting it if it carries the encryption header
    and the app is unlocked. Returns None if the file does not exist. Raises on a
    decrypt failure (wrong key / tamper) so callers can distinguish it from missing."""
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    if crypto.is_encrypted(raw):
        return crypto.decrypt_bytes(raw)      # raises crypto.AuthError if locked/wrong key
    return raw


def write_bytes(path, data: bytes):
    """Atomically write ``data``, encrypting it when the app is unlocked."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if crypto.is_unlocked():
        data = crypto.encrypt_bytes(data)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)


def read_text(path, encoding="utf-8"):
    b = read_bytes(path)
    return None if b is None else b.decode(encoding)


def write_text(path, text: str, encoding="utf-8"):
    write_bytes(path, text.encode(encoding))


def load_json(path, default):
    try:
        b = read_bytes(path)
        if b is None:
            return default
        return json.loads(b.decode("utf-8"))
    except Exception:
        return default


def save_json(path, data):
    write_bytes(path, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def load_json_plain(path, default):
    """Load JSON WITHOUT encryption. Used only for the profile registries, which must
    be readable before login (they tell the ProfileManager which profile is active)."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_plain(path, data):
    """Save JSON WITHOUT encryption (profile registries only — see load_json_plain)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def duckdb_connect(path):
    """Open a DuckDB database at ``path``, using native AES-256 encryption when the
    app is unlocked (ATTACH … ENCRYPTION_KEY). The attached DB is made the default
    schema via ``USE db`` so existing bare-table SQL needs no changes. Falls back to a
    plain connection when locked (e.g. in tests that don't exercise crypto)."""
    import duckdb
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not crypto.is_unlocked():
        return duckdb.connect(str(path))
    lit = str(path).replace("'", "''")
    con = duckdb.connect()                    # in-memory catalog; real data lives in the attached file
    con.execute(f"ATTACH '{lit}' AS db (ENCRYPTION_KEY '{crypto.dek_hex()}')")
    con.execute("USE db")
    return con


# --------------------------- Library (Resources) helpers ---------------------------

def _new_library(name="New Library"):
    """Create an empty library dict."""
    now = datetime.utcnow().isoformat()
    return {"id": uuid.uuid4().hex[:12], "name": name, "items": [], "created": now, "updated": now}


def _new_library_item(item_type="write", label="", content="", filename="", item_id=None):
    """Create a library item. type is 'write' (typed/pasted) or 'file' (imported).

    Each item carries a stable ``id`` so the RAG store can key its embeddings by
    identity rather than list position (reordering/deleting a mid-list item then no
    longer forces re-embedding of everything after it).
    """
    return {"id": item_id or uuid.uuid4().hex[:12], "type": item_type,
            "label": label, "content": content, "filename": filename}


def ensure_item_ids(lib: dict) -> bool:
    """Backfill a stable ``id`` on any library item that lacks one. Returns True if
    the library was mutated (so the caller can persist it). Idempotent."""
    changed = False
    for it in (lib or {}).get("items", []) or []:
        if not it.get("id"):
            it["id"] = uuid.uuid4().hex[:12]
            changed = True
    return changed


def _sanitize_xml_text(text: str) -> str:
    """Drop characters that are illegal in XML 1.0 so export never crashes."""
    if not text:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def library_to_xml_bytes(lib: dict) -> bytes:
    """Serialize a library dict to self-contained, pretty-printed XML bytes."""
    root = ET.Element("library")
    root.set("name", lib.get("name", ""))
    root.set("id", lib.get("id", ""))
    root.set("created", lib.get("created", ""))
    root.set("updated", lib.get("updated", ""))
    for it in lib.get("items", []):
        el = ET.SubElement(root, "item")
        if it.get("id"):
            el.set("id", it["id"])
        el.set("type", it.get("type", "write"))
        el.set("label", it.get("label", "") or "")
        if it.get("filename"):
            el.set("filename", it["filename"])
        el.text = _sanitize_xml_text(it.get("content", ""))
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")  # Python 3.9+
    except Exception:
        pass
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def library_from_xml_file(path) -> dict:
    """Parse a library XML file back into a library dict (with a fresh id)."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "library":
        raise ValueError("Not a Local Streaming Token library file (missing <library> root).")
    items = []
    for el in root.findall("item"):
        items.append(_new_library_item(
            item_type=el.get("type", "write"),
            label=el.get("label", "") or "",
            content=el.text or "",
            filename=el.get("filename", "") or "",
            item_id=el.get("id") or None,
        ))
    lib = _new_library(root.get("name") or "Imported Library")
    lib["items"] = items
    return lib


# --------------------------- Prompt library (System/Pre) helpers ---------------------------

def new_prompt_group(name="New Group"):
    return {"id": uuid.uuid4().hex[:12], "name": name, "categories": []}


def new_prompt_category(name="New Category"):
    return {"id": uuid.uuid4().hex[:12], "name": name, "prompts": []}


def new_prompt_item(name="New Prompt", prompt=""):
    return {"id": uuid.uuid4().hex[:12], "name": name, "prompt": prompt}


def prompts_to_xml_bytes(kind: str, groups: list) -> bytes:
    """Serialize a slice of a prompt tree (a list of group dicts) to self-contained,
    pretty-printed XML bytes. ``kind`` is 'system' or 'pre' so import knows the target
    tree; every exported node carries its full ancestor path (group > category > prompt)."""
    root = ET.Element("prompts")
    root.set("kind", "pre" if kind == "pre" else "system")
    for g in groups or []:
        gel = ET.SubElement(root, "group")
        gel.set("name", g.get("name", "") or "")
        for c in g.get("categories", []) or []:
            cel = ET.SubElement(gel, "category")
            cel.set("name", c.get("name", "") or "")
            for p in c.get("prompts", []) or []:
                pel = ET.SubElement(cel, "prompt")
                pel.set("name", p.get("name", "") or "")
                pel.text = _sanitize_xml_text(p.get("prompt", "") or "")
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")  # Python 3.9+
    except Exception:
        pass
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def prompts_from_xml_file(path):
    """Parse a prompt XML file into ``(kind, groups)`` with fresh ids on every node.
    ``kind`` is 'system' or 'pre'; ``groups`` is a list of group dicts."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "prompts":
        raise ValueError("Not a Local Streaming Token prompt file (missing <prompts> root).")
    kind = "pre" if (root.get("kind") or "system") == "pre" else "system"
    groups = []
    for gel in root.findall("group"):
        g = new_prompt_group(gel.get("name") or "Imported Group")
        for cel in gel.findall("category"):
            c = new_prompt_category(cel.get("name") or "Uncategorized")
            for pel in cel.findall("prompt"):
                c["prompts"].append(new_prompt_item(pel.get("name") or "Untitled", pel.text or ""))
            g["categories"].append(c)
        groups.append(g)
    return kind, groups


# --------------------------- Ollama Client ---------------------------

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def set_base(self, url: str):
        self.base_url = url.rstrip("/")

    def list_models(self):
        """Return list of model names available on this server."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=6)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch models from {self.base_url}: {e}")

    def model_capabilities(self, model: str):
        """Return the model's capability tags (e.g. ['completion', 'thinking'])
        via POST /api/show. Returns [] on any error or if the server is too old
        to report capabilities.
        """
        try:
            resp = requests.post(f"{self.base_url}/api/show", json={"model": model}, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            caps = data.get("capabilities") or []
            return [str(c).lower() for c in caps]
        except Exception:
            return []

    def embed(self, model: str, inputs):
        """Return one embedding vector (list[float]) per input string.

        Prefers the newer batch endpoint POST /api/embed
        ({"model", "input": [texts]} -> {"embeddings": [[...], ...]}); falls back to
        the older per-text POST /api/embeddings ({"model", "prompt"} -> {"embedding"})
        when the server is too old to expose /api/embed. Raises RuntimeError on
        failure so callers can degrade gracefully (fall back to full context).
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = [t if isinstance(t, str) else str(t) for t in inputs]
        if not inputs:
            return []
        # Scale the timeout with the batch: a 64-chunk request on a CPU-only host can
        # take minutes, and a spurious timeout costs the whole batch (it gets retried
        # on another server, or degrades to empty vectors).
        timeout = min(600, 60 + 2 * len(inputs))
        # Newer batch endpoint.
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": inputs},
                timeout=timeout,
            )
            if resp.status_code != 404:
                resp.raise_for_status()
                data = resp.json()
                vecs = data.get("embeddings")
                if vecs:
                    return [[float(x) for x in v] for v in vecs]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding request to {self.base_url} failed: {e}")
        # Fallback: older single-prompt endpoint, one call per input.
        out = []
        try:
            for text in inputs:
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=min(600, 60 + 2 * len(inputs)),
                )
                r.raise_for_status()
                vec = (r.json() or {}).get("embedding")
                if not vec:
                    raise RuntimeError("empty embedding returned")
                out.append([float(x) for x in vec])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding request to {self.base_url} failed: {e}")
        return out

    def complete(self, model: str, messages: list, num_ctx: int = 4096,
                 fmt=None, temperature=None, timeout: int = 180) -> str:
        """One-shot, non-streaming chat completion → assistant text.

        ``fmt`` may be a JSON schema dict (or the string "json") to request Ollama's
        structured output. Small utility for internal LLM steps (contextual chunking,
        query rewrite, memory drafting, pipeline steps) that need a single answer, not
        a token stream. Raises RuntimeError on transport failure."""
        options = {"num_ctx": int(num_ctx)}
        if temperature is not None:
            options["temperature"] = float(temperature)
        payload = {"model": model, "messages": messages, "stream": False, "options": options}
        if fmt is not None:
            payload["format"] = fmt
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Completion request to {self.base_url} failed: {e}")
        data = r.json() or {}
        return ((data.get("message") or {}).get("content") or "").strip()

    def chat_stream(self, model: str, messages: list, num_ctx: int, stop_event: threading.Event,
                    think: bool = False, tools=None, tool_executor=None, max_tool_rounds: int = 4):
        """
        Generator that yields ``(kind, text)`` tuples as they arrive, where
        ``kind`` is ``"reasoning"`` or ``"content"``. Stops early if stop_event
        is set.

        When think=True, request Ollama's reasoning mode. Ollama returns the
        model's reasoning in a separate ``message.thinking`` field, which this
        generator surfaces as ``("reasoning", …)`` chunks separate from the
        ``("content", …)`` answer chunks.

        When ``tools`` is provided, they are advertised to the model. If the
        model responds with tool calls, ``tool_executor(name, arguments)`` is
        invoked for each, the results are fed back, and generation continues —
        looping up to ``max_tool_rounds`` times before the final answer streams.
        """
        work_messages = [dict(m) for m in messages]

        for _round in range(max_tool_rounds + 1):
            payload = {
                "model": model,
                "messages": work_messages,
                "stream": True,
                "options": {"num_ctx": int(num_ctx) if num_ctx else 4096},
            }
            if think:
                payload["think"] = True
            if tools:
                payload["tools"] = tools

            assistant_content = ""
            tool_calls = []
            try:
                with requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=1800,
                ) as r:
                    r.raise_for_status()
                    for raw_line in r.iter_lines():
                        if stop_event.is_set():
                            return
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line.decode("utf-8", errors="ignore"))
                        except json.JSONDecodeError:
                            continue

                        msg = chunk.get("message") or {}
                        tcs = msg.get("tool_calls")
                        if tcs:
                            tool_calls.extend(tcs)
                        thinking = msg.get("thinking", "")
                        if thinking:
                            yield ("reasoning", thinking)
                        content = msg.get("content", "")
                        if content:
                            assistant_content += content
                            yield ("content", content)

                        if chunk.get("done"):
                            # Final frame carries the exact token counts. Surface them
                            # as a metadata tuple for the context-usage monitor; they
                            # are the source of truth over any pre-call estimate.
                            pe, ev = chunk.get("prompt_eval_count"), chunk.get("eval_count")
                            if pe is not None or ev is not None:
                                yield ("usage", {"prompt_tokens": pe or 0,
                                                 "completion_tokens": ev or 0})
                            break
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Connection error: {e}")
            except Exception as e:
                raise RuntimeError(str(e))

            # No tool calls (or nothing to run them with) -> this was the final answer.
            if not (tool_calls and tool_executor) or stop_event.is_set():
                return

            # Record the model's tool request, run each tool, feed results back.
            work_messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments")
                try:
                    result = tool_executor(name, args)
                except Exception as e:
                    result = f"Tool '{name}' failed: {e}"
                work_messages.append({"role": "tool", "name": name, "content": result})
            # loop again for the model's next turn
