#!/usr/bin/env python3
"""Message shaping — logic._clean_msg and the three provider wire formats.

Two things are being pinned down here.

The first is that adding images changed *nothing* for text: `content` is still a
plain string everywhere, and a message with no images produces byte-identical output
from every shaper. Four things downstream depend on that (the pre-prompt fold in
build_messages, both context_tracker estimators, and the batch templates), and all of
them fail quietly rather than loudly if it stops being true.

The second is that each provider gets its own multi-part shape, and only its own.
"""

import pytest

from app import logic, providers


IMG = {"media_type": "image/png", "b64": "iVBORw0KAAAA"}
IMG2 = {"media_type": "image/jpeg", "b64": "/9j/4AAQAAAA"}


def with_images(role="user", content="what is this?", imgs=(IMG,)):
    return {"role": role, "content": content, "images": list(imgs)}


# --------------------------- text-only is unchanged ---------------------------

TEXT_ONLY = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "again"},
]


def test_clean_msg_leaves_a_text_message_at_exactly_role_and_content():
    out = logic._clean_msg({"role": "user", "content": "hi", "reasoning": "secret"})
    assert out == {"role": "user", "content": "hi"}   # no stray "images" key


def test_openai_shaping_of_text_only_messages_is_unchanged():
    assert providers._openai_messages(TEXT_ONLY) == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"},
    ]


def test_anthropic_shaping_of_text_only_messages_is_unchanged():
    system, chat = providers._split_system(TEXT_ONLY)
    assert system == "You are helpful."
    assert chat == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"},
    ]


def test_ollama_shaping_of_text_only_messages_is_a_passthrough():
    out = providers._ollama_messages(TEXT_ONLY)
    assert out == TEXT_ONLY
    assert all(a is b for a, b in zip(out, TEXT_ONLY))   # not even copied


# --------------------------- images survive _clean_msg ---------------------------

def test_clean_msg_preserves_images_but_still_drops_reasoning():
    out = logic._clean_msg({"role": "user", "content": "q",
                            "images": [{"id": "img_a"}], "reasoning": "secret"})
    assert out == {"role": "user", "content": "q", "images": [{"id": "img_a"}]}


def test_clean_msg_does_not_alias_the_stored_image_list():
    stored = {"role": "user", "content": "q", "images": [{"id": "img_a"}]}
    logic._clean_msg(stored)["images"].append({"id": "img_b"})
    assert stored["images"] == [{"id": "img_a"}]


@pytest.mark.parametrize("role", ["system", "tool"])
def test_clean_msg_drops_images_from_non_conversational_roles(role):
    assert "images" not in logic._clean_msg(
        {"role": role, "content": "c", "images": [{"id": "img_a"}]})


def test_build_messages_carries_images_through_to_the_model(monkeypatch):
    """The end-to-end path: an image attached to a chat message has to survive
    build_messages, which is where every app-only field gets stripped."""
    chat = {"messages": [with_images()], "system_on": True, "system_prompt": "be terse"}
    out = logic.build_messages(chat, libraries=[])
    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[-1]["images"] == [IMG]


def test_a_pre_prompt_still_folds_into_the_text_of_an_image_turn():
    """The fold is an f-string on `content`; it would raise if content were a list."""
    chat = {"messages": [with_images()], "pre_on": True, "pre_prompt": "In French:"}
    out = logic.build_messages(chat, libraries=[])
    assert out[-1]["content"] == "In French:\n\nwhat is this?"
    assert out[-1]["images"] == [IMG]


# --------------------------- OpenAI wire format ---------------------------

def test_openai_turns_images_into_data_url_parts():
    out = providers._openai_messages([with_images(imgs=(IMG, IMG2))])
    assert out[0]["role"] == "user"
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KAAAA"
    assert parts[2]["image_url"]["url"] == "data:image/jpeg;base64,/9j/4AAQAAAA"


def test_openai_strips_images_from_a_system_turn():
    out = providers._openai_messages(
        [{"role": "system", "content": "ctx", "images": [IMG]}])
    assert out == [{"role": "system", "content": "ctx"}]


def test_openai_skips_image_entries_with_no_bytes():
    out = providers._openai_messages([with_images(imgs=({"media_type": "image/png"},))])
    assert out[0]["content"] == [{"type": "text", "text": "what is this?"}]


# --------------------------- Anthropic wire format ---------------------------

def test_anthropic_puts_images_before_the_text():
    """Claude's own guidance, and it matters for a 'what is this?' turn."""
    _system, chat = providers._split_system([with_images()])
    parts = chat[0]["content"]
    assert parts[0]["type"] == "image"
    assert parts[0]["source"] == {"type": "base64", "media_type": "image/png",
                                  "data": "iVBORw0KAAAA"}
    assert parts[1] == {"type": "text", "text": "what is this?"}


def test_anthropic_never_puts_images_on_the_system_string():
    system, chat = providers._split_system(
        [{"role": "system", "content": "ctx", "images": [IMG]}, with_images()])
    assert system == "ctx"
    assert len(chat) == 1 and chat[0]["role"] == "user"


# --------------------------- Ollama wire format ---------------------------

def test_ollama_flattens_images_to_a_bare_base64_list():
    out = providers._ollama_messages([with_images(imgs=(IMG, IMG2))])
    assert out[0]["content"] == "what is this?"     # text stays a plain string
    assert out[0]["images"] == ["iVBORw0KAAAA", "/9j/4AAQAAAA"]


def test_ollama_does_not_mutate_the_message_it_was_given():
    msg = with_images()
    providers._ollama_messages([msg])
    assert msg["images"] == [IMG]


def test_ollama_strips_images_from_a_system_turn():
    out = providers._ollama_messages(
        [{"role": "system", "content": "ctx", "images": [IMG]}])
    assert out == [{"role": "system", "content": "ctx"}]


# --------------------------- returned image parts ---------------------------

@pytest.mark.parametrize("part,expected", [
    ({"image_url": {"url": "data:image/png;base64,AAAB"}}, ("AAAB", "image/png")),
    ({"image_url": "data:image/jpeg;base64,AAAC"}, ("AAAC", "image/jpeg")),
    ({"b64_json": "AAAD"}, ("AAAD", "image/png")),
    ({"data": "AAAE", "media_type": "image/webp"}, ("AAAE", "image/webp")),
])
def test_returned_image_parts_are_parsed_in_every_shape_seen_in_the_wild(part, expected):
    assert providers._parse_image_part(part) == expected


@pytest.mark.parametrize("part", [
    None, "not a dict", {}, {"image_url": {"url": "https://example.com/a.png"}},
])
def test_an_unusable_image_part_is_ignored_rather_than_guessed_at(part):
    assert providers._parse_image_part(part) == (None, None)


def test_image_parts_are_found_in_model_extra_not_just_as_attributes():
    """The `openai` SDK types deltas strictly, so a non-standard `images` field is
    only ever reachable through model_extra — reading the attribute finds nothing."""
    class Delta:
        model_extra = {"images": [{"b64_json": "AAAF"}]}

    assert providers._extract_images(Delta()) == [{"b64_json": "AAAF"}]
    assert providers._extract_images({"images": [{"b64_json": "AAAG"}]}) == \
        [{"b64_json": "AAAG"}]
    assert providers._extract_images(None) == []
