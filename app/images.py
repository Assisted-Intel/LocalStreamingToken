#!/usr/bin/env python3
"""
Local Streaming Token — image storage and preparation.

Images are the one kind of attachment whose bytes are far too large to live inside
``chats.json``: that file is re-serialised and re-encrypted *in full* every time
``persistChat`` fires, which is 400 ms after any settings keystroke. So an image is
written once as its own encrypted file under ``core.IMAGES_DIR`` and referenced from
the chat by a short id. The record that travels over the wire carries only metadata.

The other job here is making arbitrary user files acceptable to a model. Providers
accept PNG/JPEG/GIF/WebP and nothing else, so TIFF, BMP, HEIC and friends are
transcoded; large photos are downscaled; and EXIF rotation is baked in, because a
model shown a sideways photo will confidently describe it wrong in a way that reads
as the model being stupid rather than as a bug here.

Downscaling happens lazily at send time behind an LRU cache rather than being stored,
so the full-resolution original is never lost and N parallel lanes sharing an image
pay for one decode between them.

Public API:
    SUPPORTED_EXTS / MODEL_MEDIA_TYPES / DEFAULT_MAX_DIM / THUMB_DIM
    ImageError
    is_supported(path) -> bool
    sniff(data) -> media_type ('' when unrecognised; no Pillow needed)
    load_path(path) -> (bytes, media_type)
    decode_data_url(s) -> (bytes, media_type)
    prepare(data, media_type, max_dim) -> {data, media_type, width, height,
                                           transcoded, orig_media_type, note}
    store(data, media_type, name, source, origin) -> record
    store_prepared(prep, name, source, origin) -> record
    store_file(path, max_dim, origin) -> record
    load(image_id) -> (bytes, media_type)
    exists(image_id) -> bool
    for_send(image_id, max_dim) -> (b64, media_type)
    thumb(image_id) -> (bytes, 'image/png')
    hydrate_messages(messages, max_dim) -> messages
    write_out(image_id, path) -> Path
    write_bytes_out(data, path) -> Path
    ext_for(media_type) -> '.png'
    estimate_tokens(width, height) -> int
    estimate_message_tokens(messages) -> int
    ids_in_message(msg) -> set[str]
    collect_ids(chats, batch_projects) -> set[str]
    gc(live_ids) -> int
    clear_caches()
"""

import base64
import io
import math
import re
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from . import core

# Extensions we will accept from a picker or a folder source. Anything outside
# MODEL_MEDIA_TYPES is transcoded by prepare() before it ever reaches a provider.
SUPPORTED_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".avif", ".ico", ".ppm", ".pgm", ".tga", ".jp2", ".pcx",
}
# The intersection of what Ollama, Anthropic and the OpenAI-compatible providers
# all accept. Everything else has to be converted.
MODEL_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

DEFAULT_MAX_DIM = 1568      # long-edge target: Claude's sweet spot, fine for GPT-4o/Gemini
THUMB_DIM = 256
MAX_PIXELS = 80_000_000     # decompression-bomb guard
JPEG_QUALITY = 90
# Above this, an opaque image is transcoded to JPEG instead of PNG — a 12 MP photo
# as lossless PNG is ~30 MB of base64 for no perceptual gain.
JPEG_THRESHOLD_PIXELS = 2_000_000

_HEIF_TYPES = {"image/heic", "image/heif", "image/avif"}
_ID_RE = re.compile(r"^img_[0-9a-f]{12}$")
_DATA_URL_RE = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.DOTALL)

# Files younger than this are never swept: a private chat's images exist on disk
# before the chat is ever persisted, so there is a window where nothing references
# them yet and they are still very much in use.
GC_MIN_AGE_SECONDS = 3600

# Every image id this process has minted, and the reason the age guard above is not
# enough on its own. The sweep's live set is built from the PERSISTED chats and batch
# projects, and three kinds of live image are in neither: a private chat (never
# persisted at all), an unsaved batch project's reference images, and a queued item's
# attachments. Past GC_MIN_AGE_SECONDS, deleting any chat used to unlink those — and
# silently, in both directions, since hydrate_messages drops an image it cannot load.
# Holding the ids means the sweep only ever collects what a PREVIOUS run left behind,
# which is what it was for. Cleared by clear_caches(), i.e. on a profile switch, where
# the previous profile's ids stop meaning anything.
_SESSION_IDS = set()


class ImageError(Exception):
    """Raised with a user-facing, actionable message when an image can't be used."""


# --------------------------- Pillow (lazy) ---------------------------

_heif_registered = False


def _pillow():
    """Import Pillow on first use and register the HEIF opener if it's available.

    Lazy so the app still boots (and every text-only feature still works) when
    Pillow isn't installed — same contract as ingest.py's optional parsers."""
    global _heif_registered
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ImageError("Image support needs the 'Pillow' package. "
                         "Install it with:  pip install Pillow")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    if not _heif_registered:
        _heif_registered = True
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass  # .heic then raises the actionable error below instead
    return Image, ImageOps


def is_supported(path) -> bool:
    """True if this file's extension is one we accept. Name-only check; the file
    need not exist."""
    return Path(path).suffix.lower() in SUPPORTED_EXTS


# --------------------------- Format sniffing ---------------------------

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)

_EXT_BY_TYPE = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tif",
    "image/heic": ".heic", "image/heif": ".heif", "image/avif": ".avif",
    "image/x-icon": ".ico",
}
_TYPE_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".heic": "image/heic",
    ".heif": "image/heif", ".avif": "image/avif", ".ico": "image/x-icon",
}


def sniff(data: bytes) -> str:
    """Media type from magic bytes, or '' when unrecognised. Deliberately does not
    need Pillow, so the upload route can reject junk before paying for a decode."""
    if not data:
        return ""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
            return "image/heic"
        if brand in (b"avif", b"avis"):
            return "image/avif"
    return ""


def ext_for(media_type: str) -> str:
    """File extension for a media type, defaulting to .png."""
    return _EXT_BY_TYPE.get((media_type or "").lower(), ".png")


def load_path(path):
    """Read an image file off disk -> (raw bytes, media type). Raises ImageError."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception as e:
        raise ImageError(f"Could not read {p.name}: {e}")
    if not data:
        raise ImageError(f"{p.name} is empty")
    mt = sniff(data) or _TYPE_BY_EXT.get(p.suffix.lower(), "")
    if not mt:
        raise ImageError(f"{p.name} does not look like an image")
    return data, mt


def decode_data_url(s: str):
    """'data:image/png;base64,AAAA' -> (bytes, 'image/png'). Raises ImageError."""
    m = _DATA_URL_RE.match((s or "").strip())
    if not m:
        raise ImageError("Not a base64 data URL")
    try:
        data = base64.b64decode(m.group(2), validate=False)
    except Exception as e:
        raise ImageError(f"Bad base64 image data: {e}")
    return data, (sniff(data) or m.group(1))


# --------------------------- The one transform ---------------------------

def prepare(data: bytes, media_type: str = "", max_dim: int = 0) -> dict:
    """Make ``data`` acceptable to a model, doing as little as possible.

    Returns ``{data, media_type, width, height, transcoded, orig_media_type, note}``.

    An image that is already PNG/JPEG/GIF/WebP, carries no EXIF rotation, and fits
    within ``max_dim`` is returned **byte-identical** — a pasted screenshot must not
    be silently re-encoded. Otherwise the source is decoded, EXIF orientation is
    applied, multi-frame sources collapse to frame 0, the long edge is clamped to
    ``max_dim`` (0 = no limit), and the result is written back as PNG, or JPEG when
    it is opaque and large enough that lossless would be wasteful.
    """
    media_type = (sniff(data) or media_type or "").lower()
    if not media_type:
        raise ImageError("Unrecognised image format")

    Image, ImageOps = _pillow()
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        if media_type in _HEIF_TYPES:
            raise ImageError("Reading HEIC/HEIF images needs the 'pillow-heif' package. "
                             f"Install it with:  pip install pillow-heif  ({e})")
        raise ImageError(f"Could not decode this image: {e}")

    frames = int(getattr(im, "n_frames", 1) or 1)
    # Read the orientation tag directly rather than diffing before/after pixels:
    # exif_transpose() always returns a new image, so a pixel comparison would be
    # both expensive and unable to tell "rotated" from "copied".
    try:
        needs_rotate = int((im.getexif() or {}).get(0x0112, 1) or 1) not in (0, 1)
    except Exception:
        needs_rotate = False
    if needs_rotate:
        im = ImageOps.exif_transpose(im)
    w, h = im.size
    too_big = bool(max_dim) and max(w, h) > max_dim
    known = media_type in MODEL_MEDIA_TYPES

    # Fast path: nothing to do, so hand back the exact bytes we were given.
    if known and not too_big and not needs_rotate:
        return {"data": data, "media_type": media_type, "width": w, "height": h,
                "transcoded": False, "orig_media_type": "", "note": ""}

    note = ""
    if frames > 1:
        # Only frame 0 survives a re-encode; saying so beats silently dropping pages.
        try:
            im.seek(0)
        except Exception:
            pass
        note = f"page/frame 1 of {frames} used"

    if too_big:
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)

    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    # JPEG when the source was already JPEG (it is lossy either way, and re-encoding a
    # photo as lossless PNG multiplies its size for nothing), or when a large opaque
    # image would be wasteful as PNG. Anything with transparency has to stay PNG.
    use_jpeg = not has_alpha and (
        media_type == "image/jpeg"
        or (im.size[0] * im.size[1]) > JPEG_THRESHOLD_PIXELS)
    buf = io.BytesIO()
    if use_jpeg:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
        out_type = "image/jpeg"
    else:
        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGBA" if has_alpha else "RGB")
        im.save(buf, "PNG", optimize=True)
        out_type = "image/png"

    return {"data": buf.getvalue(), "media_type": out_type,
            "width": im.size[0], "height": im.size[1],
            "transcoded": out_type != media_type,
            "orig_media_type": media_type if out_type != media_type else "",
            "note": note}


# --------------------------- Storage ---------------------------

def _new_id() -> str:
    return "img_" + uuid.uuid4().hex[:12]


def _path_for(image_id: str) -> Path:
    """On-disk path for an id. Validates the id shape because it reaches this from a
    URL segment — a bare join would be a path-traversal hole."""
    if not _ID_RE.match(image_id or ""):
        raise ImageError("Bad image id")
    return Path(core.IMAGES_DIR) / f"{image_id}.bin"


def store(data: bytes, media_type: str, name: str = "", source: str = "",
          origin: str = "user", width: int = 0, height: int = 0,
          orig_media_type: str = "") -> dict:
    """Write bytes into the active profile's image dir (encrypted, like every other
    data file) and return the record that documents reference them by."""
    image_id = _new_id()
    core.write_bytes(_path_for(image_id), data)
    # Every upload, picker result, model-generated picture and batch reference image
    # lands here, which is what makes this the one place the sweep's exemption needs.
    _SESSION_IDS.add(image_id)
    return {
        "id": image_id,
        "kind": "image",
        "name": name or ("image" + ext_for(media_type)),
        "media_type": media_type,
        "width": int(width or 0),
        "height": int(height or 0),
        "bytes": len(data),
        "orig_media_type": orig_media_type or "",
        "source": source or "",
        "origin": origin,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def store_prepared(prep: dict, name: str = "", source: str = "",
                   origin: str = "user") -> dict:
    """store() for a prepare() result — carries the dimensions and original type over."""
    return store(prep["data"], prep["media_type"], name=name, source=source,
                 origin=origin, width=prep.get("width", 0), height=prep.get("height", 0),
                 orig_media_type=prep.get("orig_media_type", ""))


def store_file(path, max_dim: int = 0, origin: str = "user") -> dict:
    """Read, prepare and store one file from disk. The picker/folder-source path."""
    p = Path(path)
    data, mt = load_path(p)
    prep = prepare(data, mt, max_dim)
    return store_prepared(prep, name=p.name, source=str(p), origin=origin)


def load(image_id: str):
    """Read stored bytes back -> (bytes, media_type). Raises ImageError when missing."""
    data = core.read_bytes(_path_for(image_id))
    if data is None:
        raise ImageError(f"Image {image_id} is no longer available")
    return data, (sniff(data) or "image/png")


def exists(image_id: str) -> bool:
    try:
        return _path_for(image_id).exists()
    except ImageError:
        return False


# --------------------------- Send-time preparation ---------------------------

@lru_cache(maxsize=64)
def _for_send_cached(image_id: str, max_dim: int):
    data, mt = load(image_id)
    if max_dim:
        prep = prepare(data, mt, max_dim)
        data, mt = prep["data"], prep["media_type"]
    return base64.b64encode(data).decode("ascii"), mt


def for_send(image_id: str, max_dim: int = 0):
    """(base64, media_type) ready for a provider, downscaled to ``max_dim``.

    Cached on (id, max_dim) and holding the *base64 string* rather than a decoded
    image, so the parallel lanes that share a pinned attachment pay for one decode
    between them instead of one each."""
    return _for_send_cached(image_id, int(max_dim or 0))


@lru_cache(maxsize=128)
def thumb(image_id: str):
    """A small PNG for chips and message bubbles -> (bytes, 'image/png')."""
    data, mt = load(image_id)
    Image, ImageOps = _pillow()
    im = Image.open(io.BytesIO(data))
    try:
        im.seek(0)          # a thumbnail of an animation is its first frame
    except Exception:
        pass
    im = ImageOps.exif_transpose(im)
    im.thumbnail((THUMB_DIM, THUMB_DIM), Image.LANCZOS)
    if im.mode not in ("RGB", "RGBA", "L", "P"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue(), "image/png"


def _clear_lru():
    """Just the decoded-bytes caches. Split out because a sweep has to drop them (an id
    it unlinked may still be cached) WITHOUT forgetting the ids this session minted."""
    _for_send_cached.cache_clear()
    thumb.cache_clear()


def clear_caches():
    """Drop the send/thumb caches and the session's minted-id set. Mandatory on a
    profile switch: the caches hold decrypted image data, so keeping them would leak the
    previous profile's pictures, and the ids belong to a store that is no longer the
    active one."""
    _clear_lru()
    _SESSION_IDS.clear()


# --------------------------- Message hydration ---------------------------

# A hard ceiling per request. Ten pinned full-resolution photos re-sent every turn
# is a runaway, and every provider has its own limit anyway.
MAX_IMAGES_PER_REQUEST = 20


def hydrate_messages(messages: list, max_dim: int = 0) -> list:
    """Replace each message's ``images`` list of records/ids with the inline
    ``{media_type, b64}`` form the adapters shape into their own wire formats.

    The one place image bytes are read during a generation. Messages without images
    are passed through untouched (same object), so the text-only path is unchanged.
    Images that no longer exist on disk are dropped rather than failing the turn.
    """
    if not messages:
        return messages
    out = []
    budget = MAX_IMAGES_PER_REQUEST
    for m in messages:
        refs = m.get("images") if isinstance(m, dict) else None
        if not refs:
            out.append(m)
            continue
        if m.get("role") not in ("user", "assistant"):
            # No provider accepts images on a system or tool turn; carrying the key
            # through would put an unrecognised field on the wire.
            msg = dict(m)
            msg.pop("images", None)
            out.append(msg)
            continue
        inline = []
        for ref in refs:
            if budget <= 0:
                break
            if isinstance(ref, str):
                ref = {"id": ref}
            if ref.get("b64"):                      # already inline
                inline.append({"media_type": ref.get("media_type") or "image/png",
                               "b64": ref["b64"]})
                budget -= 1
                continue
            try:
                b64, mt = for_send(ref.get("id", ""), max_dim)
            except ImageError:
                continue
            inline.append({"media_type": mt, "b64": b64})
            budget -= 1
        msg = dict(m)
        if inline:
            msg["images"] = inline
        else:
            msg.pop("images", None)
        out.append(msg)
    return out


def estimate_tokens(width: int, height: int) -> int:
    """Rough prompt-token cost of one image (~(w*h)/750, Anthropic's own rule of
    thumb; OpenAI's tiling lands in the same neighbourhood). Feeds the context-usage
    bar, which would otherwise report an image-only turn as costing nothing."""
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return 1100                      # unknown dimensions: assume a typical photo
    return max(1, min(1600, math.ceil((w * h) / 750)))


def estimate_message_tokens(messages: list) -> int:
    """Total estimated image cost across an assembled messages list."""
    total = 0
    for m in messages or []:
        for ref in (m.get("images") or []) if isinstance(m, dict) else []:
            if isinstance(ref, dict):
                total += estimate_tokens(ref.get("width"), ref.get("height"))
            else:
                total += estimate_tokens(0, 0)
    return total


# --------------------------- Export + lifecycle ---------------------------

def write_out(image_id: str, path) -> Path:
    """Write an image to a user-chosen path. Plain open() on purpose — this is an
    export leaving the app, not app data, so it must not be encrypted."""
    data, _mt = load(image_id)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def write_bytes_out(data: bytes, path) -> Path:
    """write_out() for bytes we already hold (a freshly returned model image)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def ids_in_message(msg: dict) -> set:
    out = set()
    for ref in (msg or {}).get("images") or []:
        if isinstance(ref, str):
            out.add(ref)
        elif isinstance(ref, dict) and ref.get("id"):
            out.add(ref["id"])
    return out


def collect_ids(chats=(), batch_projects=()) -> set:
    """Every image id referenced by the given chats and batch projects — messages,
    pinned attachments, and batch reference images."""
    live = set()
    for chat in chats or []:
        for msg in chat.get("messages") or []:
            live |= ids_in_message(msg)
        for att in chat.get("attachments") or []:
            if att.get("kind") == "image" or att.get("type") == "image":
                if att.get("id"):
                    live.add(att["id"])
            live |= ids_in_message(att)
    for proj in batch_projects or []:
        for rec in proj.get("reference_images") or []:
            if isinstance(rec, dict) and rec.get("id"):
                live.add(rec["id"])
    return live


def gc(live_ids) -> int:
    """Unlink stored images nothing references any more; returns how many went.

    A sweep rather than refcounting, because a private chat writes its images long
    before (and often instead of) ever being persisted — there is no moment at which
    a count would be correct. Files younger than GC_MIN_AGE_SECONDS are spared for
    exactly that reason: they may belong to a chat that is open right now.

    ``_SESSION_IDS`` is spared unconditionally and is the real guard: the age check
    alone only postpones the problem by an hour, and a private chat, an unsaved batch
    project or a queued item can outlive that easily. See its definition above.
    """
    live = set(live_ids or ()) | _SESSION_IDS
    removed = 0
    now = datetime.now().timestamp()
    try:
        entries = list(Path(core.IMAGES_DIR).glob("img_*.bin"))
    except Exception:
        return 0
    for p in entries:
        if p.stem in live:
            continue
        try:
            if (now - p.stat().st_mtime) < GC_MIN_AGE_SECONDS:
                continue
            p.unlink()
            removed += 1
        except Exception:
            continue
    if removed:
        _clear_lru()
    return removed
