#!/usr/bin/env python3
"""app/images.py — preparation, storage, hydration and the GC sweep.

The load-bearing behaviours here are the ones whose failure is invisible: an image
silently re-encoded (quality loss the user never asked for), a sideways photo the
model then describes wrong, or a stored image readable from the wrong data profile.
"""

import io

import pytest

from app import core, images
from conftest import isolate_paths

pytest.importorskip("PIL", reason="image tests need Pillow")


# --------------------------- fixtures / helpers ---------------------------

@pytest.fixture
def profile(tmp_path, monkeypatch):
    """An isolated, unlocked data profile so images land under tmp_path."""
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    images.clear_caches()
    yield tmp_path
    images.clear_caches()


def encode(mode="RGB", size=(64, 48), fmt="PNG", color=(200, 100, 50), **kw):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else None).save(buf, fmt, **kw)
    return buf.getvalue()


def dims(data):
    from PIL import Image
    return Image.open(io.BytesIO(data)).size


# --------------------------- sniff ---------------------------

@pytest.mark.parametrize("fmt,expected", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"),
    ("GIF", "image/gif"), ("BMP", "image/bmp"),
    ("TIFF", "image/tiff"), ("WEBP", "image/webp"),
])
def test_sniff_reads_the_magic_bytes(fmt, expected):
    assert images.sniff(encode(fmt=fmt)) == expected


def test_sniff_rejects_non_images():
    assert images.sniff(b"not an image at all") == ""
    assert images.sniff(b"") == ""


# --------------------------- prepare ---------------------------

def test_an_acceptable_image_is_returned_byte_identical():
    """The fast path. A pasted screenshot must not be silently re-encoded."""
    png = encode(fmt="PNG", size=(64, 48))
    out = images.prepare(png, "image/png", max_dim=1568)
    assert out["data"] is png or out["data"] == png
    assert out["transcoded"] is False
    assert (out["width"], out["height"]) == (64, 48)


def test_a_large_image_is_downscaled_to_the_long_edge():
    out = images.prepare(encode(fmt="JPEG", size=(4000, 2000)), "image/jpeg", max_dim=1568)
    assert max(out["width"], out["height"]) == 1568
    assert dims(out["data"]) == (1568, 784)


def test_max_dim_zero_means_full_resolution():
    out = images.prepare(encode(fmt="JPEG", size=(4000, 2000)), "image/jpeg", max_dim=0)
    assert (out["width"], out["height"]) == (4000, 2000)


def test_a_downscaled_jpeg_stays_a_jpeg():
    """Re-encoding an already-lossy photo as lossless PNG multiplies its size for
    no perceptual gain — and it is the base64 of that which fills the context."""
    out = images.prepare(encode(fmt="JPEG", size=(4000, 2000)), "image/jpeg", max_dim=1568)
    assert out["media_type"] == "image/jpeg"


@pytest.mark.parametrize("fmt,orig", [("BMP", "image/bmp"), ("TIFF", "image/tiff")])
def test_formats_no_model_accepts_are_transcoded(fmt, orig):
    out = images.prepare(encode(fmt=fmt, size=(100, 100)), max_dim=1568)
    assert out["media_type"] == "image/png"
    assert out["transcoded"] is True
    assert out["orig_media_type"] == orig


def test_transparency_survives_the_transcode():
    out = images.prepare(encode(mode="RGBA", size=(3000, 3000), fmt="PNG"),
                         "image/png", max_dim=1568)
    assert out["media_type"] == "image/png"   # never JPEG: that would flatten the alpha


def test_a_large_opaque_transcode_targets_jpeg():
    out = images.prepare(encode(fmt="TIFF", size=(3000, 3000)), max_dim=0)
    assert out["media_type"] == "image/jpeg"


def test_exif_orientation_is_baked_in():
    """Orientation 6 is 'rotate 90° CW to display' — every iPhone photo. Left alone,
    the model sees a sideways image and describes it wrong."""
    from PIL import Image
    im = Image.new("RGB", (400, 200), (10, 20, 30))
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)

    out = images.prepare(buf.getvalue(), "image/jpeg", max_dim=0)
    assert (out["width"], out["height"]) == (200, 400)   # axes swapped
    assert out["data"] != buf.getvalue()                 # not the fast path


def test_an_animated_gif_passes_through_whole_when_nothing_needs_doing():
    from PIL import Image
    frames = [Image.new("RGB", (50, 50), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    gif = buf.getvalue()

    out = images.prepare(gif, "image/gif", max_dim=1568)
    assert out["data"] == gif
    assert Image.open(io.BytesIO(out["data"])).n_frames == 2


def test_a_resized_animation_collapses_to_frame_one_and_says_so():
    from PIL import Image
    frames = [Image.new("RGB", (3000, 3000), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])

    out = images.prepare(buf.getvalue(), "image/gif", max_dim=512)
    assert "1 of 2" in out["note"]
    assert Image.open(io.BytesIO(out["data"])).n_frames == 1


def test_unrecognised_data_raises_rather_than_returning_junk():
    with pytest.raises(images.ImageError):
        images.prepare(b"definitely not an image")


def test_a_decompression_bomb_is_refused():
    from PIL import Image
    assert images.MAX_PIXELS < 100_000_000
    # Guard the constant rather than actually allocating 100 MP in a test run.
    images._pillow()
    assert Image.MAX_IMAGE_PIXELS == images.MAX_PIXELS


# --------------------------- store / load ---------------------------

def test_store_and_load_round_trip(profile):
    data = encode(fmt="PNG", size=(120, 80))
    rec = images.store(data, "image/png", name="a.png", source="C:/x/a.png",
                       width=120, height=80)

    assert rec["id"].startswith("img_")
    assert rec["kind"] == "image" and rec["origin"] == "user"
    assert rec["bytes"] == len(data)
    assert "data" not in rec and "b64" not in rec   # the record is metadata only
    assert images.load(rec["id"])[0] == data


def test_stored_bytes_are_encrypted_at_rest(profile):
    data = encode(fmt="PNG")
    rec = images.store(data, "image/png")
    on_disk = (core.IMAGES_DIR / f"{rec['id']}.bin").read_bytes()
    assert on_disk != data                          # went through core.write_bytes
    assert images.load(rec["id"])[0] == data


def test_images_do_not_leak_across_data_profiles(profile, tmp_path):
    rec = images.store(encode(fmt="PNG"), "image/png")
    core.set_active_data_profile(tmp_path / "other-profile")
    images.clear_caches()
    with pytest.raises(images.ImageError):
        images.load(rec["id"])


def test_a_malformed_id_cannot_escape_the_image_dir(profile):
    for bad in ("../../secrets", "img_zz", "", "img_" + "0" * 40):
        with pytest.raises(images.ImageError):
            images.load(bad)


def test_store_file_reads_prepares_and_records_the_source(profile, tmp_path):
    src = tmp_path / "photo.tiff"
    src.write_bytes(encode(fmt="TIFF", size=(2000, 1000)))

    rec = images.store_file(src, max_dim=1568)
    assert rec["name"] == "photo.tiff" and rec["source"] == str(src)
    assert rec["media_type"] == "image/png"      # transcoded off TIFF
    assert rec["orig_media_type"] == "image/tiff"
    assert max(rec["width"], rec["height"]) == 1568


# --------------------------- for_send / thumb ---------------------------

def test_for_send_downscales_and_is_cached(profile):
    rec = images.store_prepared(
        images.prepare(encode(fmt="JPEG", size=(4000, 2000)), "image/jpeg", 0))

    b64, media_type = images.for_send(rec["id"], 1568)
    assert media_type == "image/jpeg"
    import base64
    assert dims(base64.b64decode(b64)) == (1568, 784)
    assert images.for_send(rec["id"], 1568) == (b64, media_type)   # same tuple, cached

    full_b64, _ = images.for_send(rec["id"], 0)
    assert len(full_b64) > len(b64)


def test_thumb_is_a_small_png(profile):
    rec = images.store_prepared(images.prepare(encode(fmt="JPEG", size=(2000, 1000)),
                                               "image/jpeg", 0))
    data, media_type = images.thumb(rec["id"])
    assert media_type == "image/png"
    assert max(dims(data)) == images.THUMB_DIM


# --------------------------- hydrate_messages ---------------------------

def test_text_only_messages_are_passed_through_untouched(profile):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    out = images.hydrate_messages(msgs, 1568)
    assert out[0] is msgs[0] and out[1] is msgs[1]


def test_hydration_inlines_bytes_without_mutating_the_chat(profile):
    rec = images.store_prepared(images.prepare(encode(fmt="PNG"), "image/png", 0))
    msgs = [{"role": "user", "content": "what is this?", "images": [{"id": rec["id"]}]}]

    out = images.hydrate_messages(msgs, 1568)
    assert out[0]["images"][0]["media_type"] == "image/png"
    assert out[0]["images"][0]["b64"]
    assert msgs[0]["images"] == [{"id": rec["id"]}]   # the stored chat is unchanged


def test_images_are_stripped_from_non_conversational_turns(profile):
    rec = images.store_prepared(images.prepare(encode(fmt="PNG"), "image/png", 0))
    out = images.hydrate_messages(
        [{"role": "system", "content": "ctx", "images": [{"id": rec["id"]}]}], 0)
    assert "images" not in out[0]


def test_a_missing_image_drops_the_reference_instead_of_failing_the_turn(profile):
    out = images.hydrate_messages(
        [{"role": "user", "content": "q", "images": [{"id": "img_" + "0" * 12}]}], 0)
    assert out[0]["content"] == "q"
    assert "images" not in out[0]


def test_already_inline_images_survive_a_second_hydration(profile):
    msgs = [{"role": "user", "content": "q",
             "images": [{"media_type": "image/png", "b64": "AAAA"}]}]
    assert images.hydrate_messages(msgs, 1568)[0]["images"][0]["b64"] == "AAAA"


def test_the_per_request_image_count_is_capped(profile):
    rec = images.store_prepared(images.prepare(encode(fmt="PNG"), "image/png", 0))
    many = [{"id": rec["id"]}] * (images.MAX_IMAGES_PER_REQUEST + 5)
    out = images.hydrate_messages([{"role": "user", "content": "q", "images": many}], 0)
    assert len(out[0]["images"]) == images.MAX_IMAGES_PER_REQUEST


# --------------------------- token estimate ---------------------------

def test_an_image_is_never_estimated_as_free():
    assert images.estimate_tokens(1024, 768) > 1000
    assert images.estimate_tokens(0, 0) > 0          # unknown dimensions still cost
    assert images.estimate_tokens(9000, 9000) <= 1600


# --------------------------- gc ---------------------------

def test_gc_spares_young_files_even_when_unreferenced(profile):
    """A private chat writes its images before it is ever persisted, so 'nothing
    references this' is not yet evidence that nothing is using it."""
    rec = images.store(encode(fmt="PNG"), "image/png")
    assert images.gc(set()) == 0
    assert images.exists(rec["id"])


def _age(rec):
    """Backdate a stored image past the sweep's age guard."""
    import os
    import time
    old = time.time() - images.GC_MIN_AGE_SECONDS - 60
    os.utime(core.IMAGES_DIR / f"{rec['id']}.bin", (old, old))


def test_gc_removes_old_unreferenced_files_and_keeps_referenced_ones(profile):
    keep = images.store(encode(fmt="PNG"), "image/png")
    drop = images.store(encode(fmt="PNG", size=(20, 20)), "image/png")
    _age(keep)
    _age(drop)
    # Both were minted by this session, so both are spared until that ends.
    images.clear_caches()

    assert images.gc({keep["id"]}) == 1
    assert images.exists(keep["id"]) and not images.exists(drop["id"])


def test_gc_spares_an_image_this_session_minted_however_old_the_file_is(profile):
    """The private-chat case, and the reason the age guard alone is not enough.

    A private chat is NEVER persisted, so its images appear in no document the sweep's
    live set is built from. Past GC_MIN_AGE_SECONDS, deleting any other chat used to
    unlink them mid-conversation — silently, because hydrate_messages drops an image it
    cannot load rather than failing the turn.
    """
    rec = images.store(encode(fmt="PNG"), "image/png")
    _age(rec)

    assert images.gc(set()) == 0
    assert images.exists(rec["id"])
    # …and it stays spared however many sweeps run.
    assert images.gc(set()) == 0
    assert images.exists(rec["id"])


def test_a_sweep_does_not_forget_the_ids_this_session_minted(profile):
    """gc() drops the decoded-bytes caches when it removes something. It must not take
    the minted-id set with them, or one collected orphan would expose every live
    private-chat image to the next sweep."""
    live = images.store(encode(fmt="PNG"), "image/png")
    _age(live)
    stale = images.store(encode(fmt="PNG", size=(20, 20)), "image/png")
    _age(stale)
    images._SESSION_IDS.discard(stale["id"])   # as if a previous run had left it

    assert images.gc(set()) == 1
    assert images.exists(live["id"]) and not images.exists(stale["id"])
    assert images.gc(set()) == 0
    assert images.exists(live["id"])


def test_switching_profile_ends_the_exemption(profile):
    """clear_caches() runs on a data-profile switch. The previous profile's ids stop
    meaning anything there, so they must not keep sparing files in the new one."""
    rec = images.store(encode(fmt="PNG"), "image/png")
    _age(rec)
    images.clear_caches()

    assert images.gc(set()) == 1
    assert not images.exists(rec["id"])


def test_collect_ids_finds_every_reference_site():
    ids = images.collect_ids(
        chats=[{"messages": [{"role": "user", "images": [{"id": "img_a"}, "img_b"]}],
                "attachments": [{"type": "image", "id": "img_c"}]}],
        batch_projects=[{"reference_images": [{"id": "img_d"}]}],
    )
    assert ids == {"img_a", "img_b", "img_c", "img_d"}
