#!/usr/bin/env python3
"""The /api/images/* routes and the vision capability report.

These are the app's first and only upload endpoints, so the things worth pinning are
the ones the rest of the codebase's path-only convention never had to think about:
oversized bodies, junk masquerading as an image, and serving stored bytes back
*decrypted* (a send_file here would hand the browser ciphertext).
"""

import io

import pytest

from conftest import sse_frames, first, use_adapter, StubAdapter

pytest.importorskip("PIL", reason="image route tests need Pillow")


def encode(size=(64, 48), fmt="PNG", mode="RGB"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, (200, 100, 50)).save(buf, fmt)
    return buf.getvalue()


def upload(client, *files, max_dim=None):
    data = {"files": [(io.BytesIO(b), n) for b, n in files]}
    if max_dim is not None:
        data["max_dim"] = str(max_dim)
    return client.post("/api/images/upload", data=data,
                       content_type="multipart/form-data")


# --------------------------- upload ---------------------------

def test_upload_stores_an_image_and_returns_a_metadata_only_record(client):
    r = upload(client, (encode(), "shot.png"))
    assert r.status_code == 200
    body = r.get_json()
    assert body["errors"] == []

    rec, = body["images"]
    assert rec["id"].startswith("img_")
    assert rec["name"] == "shot.png"
    assert rec["media_type"] == "image/png"
    assert (rec["width"], rec["height"]) == (64, 48)
    assert rec["origin"] == "user"
    assert "b64" not in rec and "data" not in rec


def test_upload_accepts_several_files_at_once(client):
    r = upload(client, (encode(), "a.png"), (encode(fmt="JPEG"), "b.jpg"))
    assert [i["media_type"] for i in r.get_json()["images"]] == ["image/png", "image/jpeg"]


def test_upload_transcodes_a_format_no_model_accepts(client):
    rec, = upload(client, (encode(fmt="TIFF"), "scan.tiff")).get_json()["images"]
    assert rec["media_type"] == "image/png"
    assert rec["orig_media_type"] == "image/tiff"


def test_upload_can_downscale_on_the_way_in(client):
    rec, = upload(client, (encode(size=(4000, 2000), fmt="JPEG"), "big.jpg"),
                  max_dim=800).get_json()["images"]
    assert max(rec["width"], rec["height"]) == 800


def test_upload_keeps_the_original_resolution_by_default(client):
    """The send-time clamp is a separate decision — discarding pixels at upload
    time would be irreversible."""
    rec, = upload(client, (encode(size=(4000, 2000), fmt="JPEG"), "big.jpg")
                  ).get_json()["images"]
    assert (rec["width"], rec["height"]) == (4000, 2000)


def test_one_bad_file_does_not_cost_the_user_the_rest_of_the_drop(client):
    body = upload(client, (b"this is not an image", "notes.txt"),
                  (encode(), "good.png")).get_json()
    assert len(body["images"]) == 1 and body["images"][0]["name"] == "good.png"
    assert len(body["errors"]) == 1 and "notes.txt" in body["errors"][0]


def test_an_empty_upload_is_a_400(client):
    r = client.post("/api/images/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_an_oversized_body_gets_json_not_an_html_error_page(client):
    from app import server
    limit = server.create_app().config["MAX_CONTENT_LENGTH"]
    r = client.post("/api/images/upload",
                    data={"files": (io.BytesIO(b"x" * (limit + 1024)), "huge.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
    assert "too large" in r.get_json()["error"]


# --------------------------- fetch ---------------------------

def test_a_stored_image_is_served_back_decrypted_and_byte_identical(client):
    png = encode()
    rec, = upload(client, (png, "a.png")).get_json()["images"]

    r = client.get(f"/api/images/{rec['id']}")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert r.get_data() == png
    assert "immutable" in r.headers["Cache-Control"]


def test_the_thumb_variant_is_small(client):
    from PIL import Image
    from app import images
    rec, = upload(client, (encode(size=(2000, 1000), fmt="JPEG"), "a.jpg")
                  ).get_json()["images"]

    r = client.get(f"/api/images/{rec['id']}?thumb=1")
    assert r.mimetype == "image/png"
    assert max(Image.open(io.BytesIO(r.get_data())).size) == images.THUMB_DIM


def test_an_unknown_or_malformed_id_is_a_404_not_a_traceback(client):
    assert client.get("/api/images/img_000000000000").status_code == 404
    assert client.get("/api/images/not-an-id").status_code == 404


def test_images_require_a_login(tmp_path, monkeypatch):
    from conftest import isolate_paths
    isolate_paths(tmp_path, monkeypatch)
    from app import server
    app = server.create_app()
    app.config["TESTING"] = True
    assert app.test_client().get("/api/images/img_000000000000").status_code == 401


# --------------------------- save to disk ---------------------------

def test_save_writes_the_image_to_the_chosen_path(client, tmp_path, monkeypatch):
    from app import native_dialog
    rec, = upload(client, (encode(), "a.png")).get_json()["images"]
    dest = tmp_path / "exported.png"
    monkeypatch.setattr(native_dialog, "save_file", lambda **kw: str(dest))

    body = client.post(f"/api/images/{rec['id']}/save",
                       json={"default_name": "a.png"}).get_json()
    assert body["ok"] and body["path"] == str(dest)
    assert dest.read_bytes() == encode()          # plaintext: it left the app


def test_save_appends_the_right_extension_when_the_dialog_omits_it(client, tmp_path,
                                                                   monkeypatch):
    from app import native_dialog
    rec, = upload(client, (encode(fmt="JPEG"), "a.jpg")).get_json()["images"]
    monkeypatch.setattr(native_dialog, "save_file", lambda **kw: str(tmp_path / "out"))

    assert client.post(f"/api/images/{rec['id']}/save", json={}).get_json()["path"] \
        .endswith(".jpg")


def test_cancelling_the_save_dialog_is_not_an_error(client, monkeypatch):
    from app import native_dialog
    rec, = upload(client, (encode(), "a.png")).get_json()["images"]
    monkeypatch.setattr(native_dialog, "save_file", lambda **kw: "")

    body = client.post(f"/api/images/{rec['id']}/save", json={}).get_json()
    assert body == {"ok": False, "cancelled": True}


# --------------------------- native picker ---------------------------

def test_the_picker_stores_what_it_picked_and_reports_what_it_skipped(
        client, tmp_path, monkeypatch):
    from app import native_dialog
    good = tmp_path / "photo.png"
    good.write_bytes(encode())
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    monkeypatch.setattr(native_dialog, "pick_files", lambda **kw: [str(good), str(bad)])

    frames = sse_frames(client.post("/api/images/pick", json={}))
    done = first(frames, "complete")
    assert [i["name"] for i in done["images"]] == ["photo.png"]
    assert done["images"][0]["source"] == str(good)
    assert len(done["errors"]) == 1 and "notes.txt" in done["errors"][0]


def test_cancelling_the_picker_yields_an_empty_result(client, monkeypatch):
    from app import native_dialog
    monkeypatch.setattr(native_dialog, "pick_files", lambda **kw: [])
    done = first(sse_frames(client.post("/api/images/pick", json={})), "complete")
    assert done == {"images": [], "errors": []}


# --------------------------- capabilities ---------------------------

CLOUD_URL = "https://api.openai.com/v1"
OLLAMA_URL = "http://127.0.0.1:11434"


@pytest.fixture
def cloud(client):
    """A client with a real cloud server configured. Needed because an unregistered
    base_url resolves to Ollama, which would send every lookup down the
    capability-tag path instead of the name-heuristic one."""
    r = client.put("/api/servers", json={"servers": [
        {"name": "OpenAI", "type": "openai", "base_url": CLOUD_URL, "api_key": "k"}]})
    assert r.status_code == 200, r.get_data(as_text=True)
    return client


def caps(client, model, server=CLOUD_URL):
    return client.get(f"/api/models/capabilities?server={server}&model={model}").get_json()


@pytest.mark.parametrize("model", ["gpt-4o", "claude-sonnet-5", "gemini-2.5-flash",
                                   "qwen2.5-vl", "llava-next", "pixtral-12b"])
def test_known_vision_models_report_true(cloud, model):
    assert caps(cloud, model)["vision"] is True


@pytest.mark.parametrize("model", ["deepseek-chat", "gpt-3.5-turbo", "o1-mini"])
def test_known_text_only_models_report_false(cloud, model):
    assert caps(cloud, model)["vision"] is False


def test_an_unrecognised_cloud_model_reports_unknown_rather_than_guessing(cloud):
    """None keeps the UI quiet. Guessing False would warn about a capable model;
    guessing True would stay silent about an incapable one."""
    assert caps(cloud, "some-new-model-v9")["vision"] is None


def test_vision_comes_from_the_real_tag_on_an_ollama_server(client, monkeypatch):
    class VisionAdapter(StubAdapter):
        def model_capabilities(self, model):
            return ["completion", "vision"]

    use_adapter(monkeypatch, VisionAdapter())
    assert caps(client, "anything-at-all", server=OLLAMA_URL)["vision"] is True


def test_an_ollama_model_without_the_tag_reports_no_vision(client, monkeypatch):
    use_adapter(monkeypatch, StubAdapter())          # model_capabilities -> []
    assert caps(client, "llama3", server=OLLAMA_URL)["vision"] is None


def test_image_output_is_reported_only_for_the_models_that_do_it(cloud):
    assert caps(cloud, "gemini-2.5-flash-image")["image_output"] is True
    assert caps(cloud, "gpt-4o")["image_output"] is False


def test_ollama_never_claims_image_output(client):
    """Its chat endpoint reads images but has no way to return one."""
    assert caps(client, "gemini-2.5-flash-image", server=OLLAMA_URL)["image_output"] is False


# --------------------------- end to end through generation ---------------------------
# The chain an attached image actually travels: chat dict -> logic.build_messages ->
# images.hydrate_messages -> adapter. Every link in it strips unknown message fields
# for good reasons, so this is the test that catches a regression anywhere along it.

def send(client, chat, adapter):
    r = client.post(f"/api/chats/{chat['id']}/send",
                    json={"chat": chat, "run_id": "test-run"})
    return sse_frames(r)


def chat_with_image(client, image_id, **extra):
    chat = client.post("/api/chats", json={"model": "test-model"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "what is this?",
                         "images": [{"id": image_id}]}]
    chat.update(extra)
    return chat


def test_an_attached_image_reaches_the_model_as_inline_bytes(client, monkeypatch):
    png = encode()
    rec, = upload(client, (png, "a.png")).get_json()["images"]
    adapter = use_adapter(monkeypatch, StubAdapter(["a red square"]))

    send(client, chat_with_image(client, rec["id"]), adapter)

    _model, messages, _opts = adapter.seen[0]
    user = messages[-1]
    assert user["content"] == "what is this?"      # text stayed a plain string
    assert user["images"][0]["media_type"] == "image/png"
    assert user["images"][0]["b64"]                # ids were swapped for bytes


def test_the_send_time_clamp_shrinks_a_large_image(client, monkeypatch):
    import base64
    from PIL import Image
    rec, = upload(client, (encode(size=(4000, 2000), fmt="JPEG"), "big.jpg")
                  ).get_json()["images"]
    client.post("/api/settings", json={"image_max_dim": 800})
    adapter = use_adapter(monkeypatch, StubAdapter(["ok"]))

    send(client, chat_with_image(client, rec["id"]), adapter)

    sent = adapter.seen[0][1][-1]["images"][0]["b64"]
    assert max(Image.open(io.BytesIO(base64.b64decode(sent))).size) == 800


def test_full_res_opts_out_of_the_clamp(client, monkeypatch):
    import base64
    from PIL import Image
    rec, = upload(client, (encode(size=(4000, 2000), fmt="JPEG"), "big.jpg")
                  ).get_json()["images"]
    client.post("/api/settings", json={"image_max_dim": 800})
    adapter = use_adapter(monkeypatch, StubAdapter(["ok"]))

    send(client, chat_with_image(client, rec["id"], image_full_res=True), adapter)

    sent = adapter.seen[0][1][-1]["images"][0]["b64"]
    assert max(Image.open(io.BytesIO(base64.b64decode(sent))).size) == 4000


def test_an_image_turn_is_not_reported_as_costing_nothing(client, monkeypatch):
    """content is "" for an image-only turn, so the text estimator sees zero. The bar
    would silently under-report by thousands of tokens without the image addend."""
    rec, = upload(client, (encode(size=(1024, 768)), "a.png")).get_json()["images"]
    adapter = use_adapter(monkeypatch, StubAdapter(["ok"]))
    chat = chat_with_image(client, rec["id"])
    chat["messages"][0]["content"] = ""

    frames = send(client, chat, adapter)
    started = next(d for e, d in frames if e == "context" and d["phase"] == "start")
    assert started["prompt_tokens"] > 500
    assert started["breakdown"]["user"] > 500


def test_a_returned_image_is_stored_and_streamed_as_a_record(client, monkeypatch):
    import base64
    png = encode()
    adapter = use_adapter(monkeypatch, StubAdapter([[
        ("content", "here you go"),
        ("image", {"b64": base64.b64encode(png).decode(), "media_type": "image/png",
                   "index": 0}),
    ]]))
    chat = client.post("/api/chats", json={"model": "test-model"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "draw a red square"}]

    frames = send(client, chat, adapter)
    rec = first(frames, "image")
    assert rec["origin"] == "model"
    assert "b64" not in rec                       # the id crosses the wire, not the bytes
    assert client.get(f"/api/images/{rec['id']}").get_data() == png
    assert first(frames, "pass_end")["images"] == [rec]


def test_an_undecodable_returned_image_does_not_take_the_answer_with_it(client,
                                                                        monkeypatch):
    adapter = use_adapter(monkeypatch, StubAdapter([[
        ("image", {"b64": "!!!not base64!!!", "media_type": "image/png"}),
        ("content", "the text still arrives"),
    ]]))
    chat = client.post("/api/chats", json={"model": "test-model"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "draw"}]

    frames = send(client, chat, adapter)
    assert not [d for e, d in frames if e == "image"]
    assert first(frames, "pass_end")["content"] == "the text still arrives"


# --------------------------- settings ---------------------------

def test_the_image_settings_persist(client):
    r = client.post("/api/settings", json={"image_max_dim": 2048,
                                           "image_full_res_default": True})
    cfg = r.get_json()["config"]
    assert cfg["image_max_dim"] == 2048 and cfg["image_full_res_default"] is True


def test_an_absurd_max_dim_is_clamped_rather_than_stored(client):
    assert client.post("/api/settings", json={"image_max_dim": 99999}
                       ).get_json()["config"]["image_max_dim"] == 8192
    assert client.post("/api/settings", json={"image_max_dim": 1}
                       ).get_json()["config"]["image_max_dim"] == 256


# --------------------------- lifecycle ---------------------------

def test_an_export_embeds_the_images_its_chats_reference(client, tmp_path, monkeypatch):
    from app import native_dialog
    png = encode()
    rec, = upload(client, (png, "a.png")).get_json()["images"]
    chat = client.post("/api/chats", json={"model": "m"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "q", "images": [{"id": rec["id"]}]}]
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})

    dest = tmp_path / "export.json"
    monkeypatch.setattr(native_dialog, "save_file", lambda **kw: str(dest))
    body = client.post("/api/chats/export", json={"scope": "all"}).get_json()
    assert body["images"] == 1        # so the UI can warn it isn't encrypted

    import base64
    import json as _json
    envelope = _json.loads(dest.read_text(encoding="utf-8"))
    assert base64.b64decode(envelope["images"][rec["id"]]["data"]) == png


def test_importing_rewrites_image_ids_to_locally_stored_copies(client, tmp_path,
                                                               monkeypatch):
    """Ids are minted per store, so an envelope's ids mean nothing here. Reusing them
    verbatim would either 404 or, worse, collide with an unrelated local image."""
    from app import native_dialog
    png = encode()
    rec, = upload(client, (png, "a.png")).get_json()["images"]
    chat = client.post("/api/chats", json={"model": "m"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "q", "images": [{"id": rec["id"]}]}]
    chat["attachments"] = [{"id": rec["id"], "type": "image", "label": "a.png",
                            "content": ""}]
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})

    dest = tmp_path / "export.json"
    monkeypatch.setattr(native_dialog, "save_file", lambda **kw: str(dest))
    client.post("/api/chats/export", json={"scope": "all"})
    monkeypatch.setattr(native_dialog, "pick_files", lambda **kw: [str(dest)])
    r = client.post("/api/chats/import", json={})
    assert r.status_code == 200, r.get_data(as_text=True)

    imported = [c for c in r.get_json()["chats"] if c["id"] != chat["id"]]
    assert imported, "nothing was imported"
    full = client.get(f"/api/chats/{imported[0]['id']}").get_json()["chat"]
    new_id = full["messages"][0]["images"][0]["id"]
    assert new_id != rec["id"]
    assert full["attachments"][0]["id"] == new_id     # both reference sites remapped
    assert client.get(f"/api/images/{new_id}").get_data() == png


def test_deleting_a_chat_does_not_take_a_live_chats_images_with_it(client):
    """gc runs on delete; the age guard is what stops it gutting an open chat."""
    keep, = upload(client, (encode(), "keep.png")).get_json()["images"]
    drop, = upload(client, (encode(size=(20, 20)), "drop.png")).get_json()["images"]
    chat = client.post("/api/chats", json={"model": "m"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "q", "images": [{"id": drop["id"]}]}]
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})

    assert client.delete(f"/api/chats/{chat['id']}").status_code == 200
    assert client.get(f"/api/images/{keep['id']}").status_code == 200
    assert client.get(f"/api/images/{drop['id']}").status_code == 200


def test_switching_data_profiles_drops_the_decrypted_image_cache(client):
    """for_send/thumb cache decrypted bytes keyed only by id. Carrying that across a
    profile switch would serve the previous profile's pictures."""
    from app import images
    rec, = upload(client, (encode(), "a.png")).get_json()["images"]
    images.clear_caches()          # the lru_caches are process-global across tests
    assert client.get(f"/api/images/{rec['id']}?thumb=1").status_code == 200
    assert images.thumb.cache_info().currsize == 1

    r = client.post("/api/profiles/data", json={"name": "Other"})
    assert r.status_code == 200, r.get_data(as_text=True)
    other = next(p["id"] for p in r.get_json()["profiles"]["data"]["profiles"]
                 if p["name"] == "Other")
    assert client.post(f"/api/profiles/data/{other}/activate").status_code == 200

    assert images.thumb.cache_info().currsize == 0
    assert client.get(f"/api/images/{rec['id']}").status_code == 404


@pytest.mark.parametrize("mode", ["new", "merge"])
def test_saving_an_incognito_session_carries_its_images_across(client, mode):
    """An image written during a private session lives in the scratch tree, which is
    about to be discarded. Without the copy, every saved chat would point at a
    picture that no longer exists."""
    assert client.post("/api/profiles/data/incognito",
                       json={"seed": "blank"}).status_code == 200
    png = encode()
    rec, = upload(client, (png, "private.png")).get_json()["images"]
    chat = client.post("/api/chats", json={"model": "m"}).get_json()["chat"]
    chat["messages"] = [{"role": "user", "content": "q", "images": [{"id": rec["id"]}]}]
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})

    if mode == "new":
        body = {"mode": "new", "name": "Saved Session"}
    else:
        target = client.get("/api/profiles").get_json()["profiles"]["data"]["profiles"][0]
        body = {"mode": "merge", "target_id": target["id"]}
    r = client.post("/api/profiles/data/incognito/save", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)

    saved = client.get("/api/chats").get_json()["chats"]
    assert saved, "the private chat was not saved"
    full = client.get(f"/api/chats/{saved[0]['id']}").get_json()["chat"]
    image_id = full["messages"][0]["images"][0]["id"]
    assert client.get(f"/api/images/{image_id}").get_data() == png
